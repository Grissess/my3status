import json, sys, time, os, colorsys, re, multiprocessing, asyncio, concurrent.futures, traceback, math, ctypes, signal

import netifaces, psutil, ddate.base

def get_loop(loop = None):
    if loop is None:
        return asyncio.get_running_loop()
    return loop

class SleepWaiter(object):
    def __init__(self, interval):
        self.interval = interval

    async def wait(self):
        await asyncio.sleep(self.interval)

class SleepBiasWaiter(object):
    def __init__(self, interval, bias = 0.0, corr = 0.1, icorr = 0.5, minint = 0.25, clock = time.CLOCK_REALTIME):
        self.interval, self.bias, self.clock = interval, bias, clock
        self.corr, self.icorr, self.minint = corr, icorr, minint
        self.bias_corr = 0.0
        self.interval_corr = 1.0
        self.reset = True
        self.start = None

    async def wait(self):
        if self.start is not None:
            now = self.start
            if not self.reset:
                dt = (now - self.bias) % self.interval / self.interval
                if dt >= 0.5:
                    dt -= 1.0
                delta_bias = self.corr * dt
                self.bias_corr += delta_bias
                self.bias_corr %= self.interval
                self.interval_corr *= 1.0 - (self.icorr * delta_bias)
                #print('dt:', dt, 'db:', delta_bias, 'bias_corr:', self.bias_corr, 'interval_corr:', self.intval_corr, file=sys.stderr)
            else:
                self.bias_corr = 0.0
                self.interval_corr = 1.0
            self.reset = False
            ci = self.interval_corr * self.interval
            dur = ci * (1.0 - (now / self.interval - int(now / self.interval)))
            if dur < self.minint * ci:
                # Usually because we missed a goal slightly early due to jitter
                dur += ci
            #print('Sleep:', dur, file=sys.stderr)
            await asyncio.sleep(dur)
        self.start = time.clock_gettime(self.clock)  # this is the point we want to sync to bias

class Timespec(ctypes.Structure):
    _fields_ = [('sec', ctypes.c_long), ('nsec', ctypes.c_long)]
class ITimerSpec(ctypes.Structure):
    _fields_ = [('interval', Timespec), ('value', Timespec)]
p_ITimerSpec = ctypes.POINTER(ITimerSpec)
class PosixTimerWaiter(object):
    f_timer_create = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
            use_errno = True,
    )
    pi_timer_create = (
            (1, 'clockid', time.CLOCK_REALTIME),
            (1, 'evp', None),
            (2, 'timerid'),
    )
    f_timer_settime = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int, p_ITimerSpec, p_ITimerSpec,
            use_errno = True,
    )
    pi_timer_settime = (
            (1, 'timerid'),
            (1, 'flags', 0),
            (1, 'new_value'),
            (2, 'old_value'),
    )
    TIMER_ABSTIME = 1

    def __init__(self, interval, clock = time.CLOCK_REALTIME):
        self.interval = interval
        self.lib = ctypes.CDLL('librt.so.1')
        self.timer_create = self.f_timer_create(('timer_create', self.lib), self.pi_timer_create)
        self.timer_settime = self.f_timer_settime(('timer_settime', self.lib), self.pi_timer_settime)
        self.clock = clock
        self.timer = self.timer_create(clock)
        async def open_gate(cond):
            async with cond:
                cond.notify(1)
        def on_alarm(loop, coro, args):
            asyncio.run_coroutine_threadsafe(coro(*args), loop)
        self.signal_setup = (open_gate, on_alarm)

    def install_sighand(self):
        if self.signal_setup is not None:
            loop = asyncio.get_event_loop()
            self.gate = asyncio.Condition()
            open_gate, on_alarm = self.signal_setup
            loop.add_signal_handler(signal.SIGALRM, on_alarm, loop, open_gate, (self.gate,))
            self.signal_setup = None

    async def wait(self):
        self.install_sighand()
        now = time.clock_gettime(self.clock)
        nxt = math.ceil(now / self.interval) * self.interval
        nxs = int(nxt)
        nxns = int(1e9 * (nxt - nxs))
        self.timer_settime(
                self.timer,
                self.TIMER_ABSTIME,
                ITimerSpec((0, 0), (nxs, nxns)),
        )
        async with self.gate:
            await self.gate.wait()

class Status(object):
    def __init__(self, *providers, waiter = SleepBiasWaiter(0.5)):
        self.providers = list(providers)
        self.idmap = {id(provider): provider for provider in providers}
        self.reschedule()
        self.waiter = waiter
        self.stop = True
        # We only spawn two coroutines; the third is for any necessary internal
        # tasks
        self.executor = concurrent.futures.ThreadPoolExecutor(3)

    def reschedule(self):
        self.prov_order = sorted(enumerate(self.providers), key = lambda pr: pr[1].priority, reverse = True)

    def add(self, provider):
        self.providers.append(provider)
        self.idmap[id(provider)] = provider
        self.reschedule()

    async def awrite(self, f, data, loop = None):
        return await get_loop(loop).run_in_executor(self.executor, f.write, data)

    async def aflush(self, f, loop = None):
        return await get_loop(loop).run_in_executor(self.executor, f.flush)

    async def aread(self, f, bufsz, loop = None):
        return await get_loop(loop).run_in_executor(self.executor, f.read, bufsz)

    async def areadline(self, f, loop = None):
        return await get_loop(loop).run_in_executor(self.executor, f.readline)

    async def co_output(self, fo):
        await self.awrite(fo, json.dumps({'version': 1, 'click_events': True}))
        await self.awrite(fo, '\n[\n')
        while not self.stop:
            op = [None] * len(self.providers)
            for i, p in self.prov_order:
                op[i] = p.process()
            await self.awrite(fo, json.dumps(op))
            await self.awrite(fo, ',')
            await self.aflush(fo)
            self.waiter.start_time = time.clock_gettime(time.CLOCK_REALTIME)
            await self.waiter.wait()
            self.waiter.end_time = time.clock_gettime(time.CLOCK_REALTIME)
        await self.awrite(fo, '\n]\n')

    async def co_input(self, fi):
        #log = open('/tmp/my3status.log', 'w')
        while not self.stop:
            line = (await self.areadline(fi)).strip().lstrip(',')
            if line == '[':  # Beginning of the stream
                continue
            if line == ']':  # ... end of the stream?!
                break
            try:
                block = json.loads(line)
                inst = self.idmap[int(block['instance'])]
                inst.click(block)
            except Exception:
                await self.awrite(log, traceback.format_exc())
                await self.aflush(log)

    async def co_run(self):
        return await asyncio.gather(
                self.co_input(sys.stdin),
                self.co_output(sys.stdout),
        )

    def run(self):
        self.stop = False
        return asyncio.run(self.co_run())

class Provider(object):
    color = '#ffffff'
    cached = {}
    priority = 0

    def run_common(self, short = False):
        return {'color': self.color, 'name': type(self).__name__, 'instance': id(self), 'markup': 'pango'}

    def run(self):
        return self.run_common(False)

    def run_short(self):
        return self.run_common(True)

    interval = None
    last_run = None
    def should_run(self):
        if self.interval is None:
            return True
        if self.last_run is not None and self.last_run + self.interval > time.time():
            return False
        self.last_run = time.time()
        return True

    short = True
    def process(self):
        if self.should_run():
            try:
                self.cached = self.run_short() if self.short else self.run()
            except Exception as e:
                self.cached = self.err_block(e)
        return self.cached

    def err_block(self, exc):
        return {'color': '#ff00ff', 'full_text': str(exc)}

    def click(self, block):
        if block['button'] == 1:
            self.short = not self.short
            return True
        return False

    low_hue = 0.0
    high_hue = 2.0/3.0
    low_value = 0.0
    high_value = 100.0

    def get_gradient(self, v,
                     l=None, h=None,
                     lh=None, hh=None,
                     ls=1.0, hs=1.0,
                     lv=1.0, hv=1.0):
        if l is None:
            l = self.low_value
        if h is None:
            h = self.high_value
        if lh is None:
            lh = self.low_hue
        if hh is None:
            hh = self.high_hue
        clamped = min((h, max((l, v))))
        ratio = (clamped - l) / (h - l)
        lerp = lambda l, h, ratio=ratio: ratio * h + (1 - ratio) * l
        r, g, b = colorsys.hsv_to_rgb(lerp(lh, hh), lerp(ls, hs), lerp(lv, hv))
        r, g, b = tuple(min((int(i*256), 255)) for i in (r, g, b))
        return '#{:02x}{:02x}{:02x}'.format(r, g, b)

    BARS = [' ', '▏', '▎', '▍', '▌', '▋', '▋', '▊', '▊', '█']
    BLOCK = '█' 

    def get_bar(self, v, l=0.0, h=1.0, bg=None, fg=None):
        p = int(100.0 * (v - l) / (h - l))
        t, o = divmod(p, 10)
        if o == 0:
            b = (self.BLOCK * t).ljust(10)
        else:
            b = (self.BLOCK * t + self.BARS[o]).ljust(10)
        fgs = '' if fg is None else f' foreground="{fg}"'
        bgs = '' if bg is None else f' background="{bg}"'
        if not (fgs or bgs):
            return b
        return f'<span{fgs}{bgs}>{b}</span>'

    VERTS = ' _▁▂▃▄▅▆▇█'

    def get_vert_bar(self, v, l=0.0, h=1.0, bg='#222222', fg=None):
        n = (v - l) / (h - l)
        c = self.VERTS[min((len(self.VERTS) - 1, max((0, int(n * len(self.VERTS))))))]
        fgs = '' if fg is None else f' foreground="{fg}"'
        bgs = '' if bg is None else f' background="{bg}"'
        if not (fgs or bgs):
            return c
        return f'<span{fgs}{bgs}>{c}</span>'

class DiskProvider(Provider):
    crit_free = 1024**3
    format = '{path}:{free:.2f}G{vert_bar}'
    format_short = '{path}:{vert_bar}'
    div = 1024**3

    def __init__(self, path):
        self.path = path

    def run_common(self, short = False):
        vfs = os.statvfs(self.path)

        free = vfs.f_bfree * vfs.f_bsize
        size = vfs.f_blocks * vfs.f_bsize
        partial = free / size

        fd = {
            'path': self.path,
            'total': size / self.div,
            'free': free / self.div,
            'percent': 100.0 * partial,
            'bar': self.get_bar(partial),
            'vert_bar': self.get_vert_bar(partial),
        }

        block = super().run_common(short)
        block['full_text'] = (self.format_short if short else self.format).format(**fd)
        block['urgent'] = free < self.crit_free
        return block

class NetworkProvider(Provider):
    format_down = '!{interface}'
    color_down = '#333333'
    format_up = '{addr}/{bits}'
    color_up = '#777777'
    format_multiple = '{addr}/{bits}({num}/{total})'
    color_multiple = '#77cc77'
    cycle_period = 2

    hide = re.compile(r'lo|sit.*|ip6tnl.*|.*\.\d+')
    af_priority = (netifaces.AF_INET, netifaces.AF_INET6)
    sep = ' '

    ENDMASK = re.compile(r'.*/(\d+)$')

    @staticmethod
    def bits_in_ipv4(dd):
        return sum(bin(int(c)).count('1') for c in dd.split('.'))

    def run(self):
        texts = []

        for iface in netifaces.interfaces():
            if self.hide.match(iface):
                continue

            addrs = netifaces.ifaddresses(iface)
            for af in self.af_priority:
                data = addrs.get(af)
                if not data:
                    continue
                if len(data) == 1:
                    data = data[0]
                    data['interface'] = iface
                    data['bits'] = self.bits_in_ipv4(data['netmask']) if af == netifaces.AF_INET else int(self.ENDMASK.match(data['netmask']).group(1))
                    texts.append('<span foreground="{}">{}</span>'.format(self.color_up, self.format_up.format(**data)))
                else:
                    size = len(data)
                    index = int(time.time() / self.cycle_period) % size
                    data = data[index]
                    data['num'] = index + 1
                    data['total'] = size
                    data['interface'] = iface
                    data['bits'] = self.bits_in_ipv4(data['netmask']) if af == netifaces.AF_INET else int(self.ENDMASK.match(data['netmask']).group(1))
                    texts.append('<span foreground="{}">{}</span>'.format(self.color_multiple, self.format_multiple.format(**data)))
                break
            else:
                texts.append('<span foreground="{}">{}</span>'.format(self.color_down, self.format_down.format(interface=iface)))

        block = super().run()
        block['full_text'] = self.sep.join(texts)
        block['markup'] = 'pango'
        return block

    def run_short(self):
        block = super().run_short()
        afs = set(self.af_priority)
        block['color'] = (self.color_up if
                any(set(netifaces.ifaddresses(iface).keys()) & afs for iface in netifaces.interfaces() if not self.hide.match(iface))
                else self.color_down
        )
        block['full_text'] = 'NET'
        return block

class TemperatureProvider(Provider):
    low_value = 20.0
    low_hue = 2.0/3.0
    high_value = 70.0
    high_hue = 0.0
    crit_temp = 80.0
    format = '{tempmg:.0f}mG'

    def __init__(self, path='/sys/class/thermal/thermal_zone0/temp'):
        self.path = path
        self.f = open(self.path)

    def run_common(self, short = False):
        self.f.seek(0)
        temp = float(self.f.read()) / 1000.0

        block = super().run_common(short)
        block['full_text'] = self.format.format(temp=temp, tempmg=temp*10)
        block['color'] = self.get_gradient(temp)
        block['urgent'] = temp >= self.crit_temp
        return block

    @classmethod
    def all(cls, root='/sys/class/thermal/'):
        for ent in sorted(os.listdir(root)):
            if not ent.startswith('thermal_zone'):
                continue
            path = os.path.join(root, ent, 'temp')
            if not os.path.exists(path):
                continue
            yield cls(path)


class BatteryProvider(Provider):
    format = '{status} {remaining} {flow} {voltage} {percentage:.2f}% {bar}'
    format_short = '{status}{percentage:.2f}%{vert_bar}'

    missing_color = '#ff00ff'
    missing_urgent = False

    low_value = 0.0
    high_value = 1.0
    crit_value = 0.05

    voltage_high = 12.5
    voltage_low_hue = 0.0
    voltage_high_hue = 2.0/3.0

    STATUSES = {
        'Discharging': '↓',
        'Charging': '↑',
        'Depleted': '!',
        'Full': '→',
        'Unknown': '',
    }
    MICRO_UNITS = re.compile(r'VOLTAGE|CHARGE|CURRENT|POWER|ENERGY')

    def __init__(self, battery='BAT0'):
        self.battery = battery
        self.f = None

    @staticmethod
    def file_to_dict(f):
        ret = {}

        for ln in f:
            l, sep, r = ln.partition('=')
            if not sep:
                continue
            ret[l] = r.strip()

        return ret

    def block_missing(self):
        block = super().run_common(False)
        block['full_text'] = '{battery} MISSING'.format(battery=self.battery)
        block['color'] = self.missing_color
        block['urgent'] = self.missing_urgent
        return block

    def get_info(self):
        if self.f is None:
            try:
                self.f = open('/sys/class/power_supply/{}/uevent'.format(self.battery))
            except OSError:
                return None

        self.f.seek(0)
        # Strip leading POWER_SUPPLY_
        try:
            info = {k[13:]: v for k,v in self.file_to_dict(self.f).items()}
        except OSError:
            self.f = None
            return self.block_missing()

        for k in list(info.keys()):
            if self.MICRO_UNITS.match(k):
                info[k] = float(info[k]) / 1000000.0

        info['status'] = self.STATUSES.get(info['STATUS'], info['STATUS'])
        h, m = None, None
        if 'ENERGY_NOW' in info:
            partial = info['ENERGY_NOW'] / info['ENERGY_FULL']
            info['flow'] = '{:.2f}W'.format(info['POWER_NOW'])
            energy_target = info['ENERGY_NOW'] if info['STATUS'] == 'Discharging' else (info['ENERGY_FULL'] - info['ENERGY_NOW'])
            if energy_target > 0 and info['POWER_NOW'] > 0:
                h, m = divmod(energy_target / info['POWER_NOW'] * 60.0, 60.0)
        else:
            partial = info['CHARGE_NOW'] / info['CHARGE_FULL']
            info['flow'] = '{:.3f}A'.format(info['CURRENT_NOW'])
            energy_target = info['CHARGE_NOW'] if info['STATUS'] == 'Discharging' else (info['CHARGE_FULL'] - info['CHARGE_NOW'])
            if energy_target > 0 and info['CURRENT_NOW'] > 0:
                h, m = divmod(energy_target / info['CURRENT_NOW'] * 60.0, 60.0)
        info['remaining'] = '{}h{:02}m'.format(int(h), int(m)) if h is not None else ''
        vmin, vnow = info['VOLTAGE_MIN_DESIGN'], info['VOLTAGE_NOW']
        info['voltage'] = '<span foreground="{}">{}V</span>'.format(
            self.get_gradient(vnow, l=vmin, h=self.voltage_high, lh=self.voltage_low_hue, hh=self.voltage_high_hue),
            vnow,
        )
        
        info['bar'] = self.get_bar(partial)
        info['vert_bar'] = self.get_vert_bar(partial)
        info['percentage'] = 100.0 * partial
        info['partial'] = partial

        return info

    def run_common(self, short = False):
        info = self.get_info()
        if info is None:
            return self.block_missing()
        partial = info['partial']
        block = super().run_common(short)
        block['full_text'] = (self.format_short if short else self.format).format(**info)
        block['color'] = self.get_gradient(partial)
        block['urgent'] = partial <= self.crit_value and info['STATUS'] == 'Discharging'
        block['markup'] = 'pango'
        return block

class LoadAverageProvider(Provider):
    path = '/proc/loadavg'
    format = '{avg1} {avg5} {avg15}'
    format_short = '{avg1}'
    crit_load = float(multiprocessing.cpu_count())

    def __init__(self, path=None):
        if path is not None:
            self.path = path
        self.f = open(self.path)

    def run_common(self, short = False):
        self.f.seek(0)
        avg1, avg5, avg15, tasks, lastpid = self.f.read().split(' ', 5)

        block = super().run_common(short)
        block['full_text'] = (self.format_short if short else self.format).format(avg1=avg1, avg5=avg5, avg15=avg15, tasks=tasks)
        block['urgent'] = float(avg1) >= self.crit_load
        return block

class CPUBarProvider(Provider):
    path = '/proc/stat'

    low_hues = [2.0/3.0, 2.0/3.0, 1.0, 2.0/3.0, 1.0/3.0, 1.0/3.0, 0.0, 1.0/2.0, 1.0/2.0]
    high_hues = [5.0/6.0, 5.0/6.0, 5.0/6.0, 1.0/2.0, 1.0/6.0, 1.0/6.0, 0.0, 1.0/2.0, 1.0/2.0]
    low_sats = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    high_sats = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    low_vals = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0]
    high_vals = [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0]
    bgs = ['#000011', '#000011', '#110000', '#110011', '#001111', '#001100', '#001100', '#110000', '#001111', '#001111']

    def __init__(self):
        self.last_total = 0
        self.last = [0, 0, 0, 0, 0, 0, 0]
        self.f = open(self.path)

    def run_common(self, short = False):
        self.f.seek(0)
        parts = list(map(int, self.f.readline().strip().split()[1:]))

        total = sum(parts)
        total_delta = total - self.last_total
        self.last_total = total

        delta = [now - last for now, last in zip(parts, self.last)]
        fraction = [d / total_delta for d in delta]
        self.last = parts

        if len(fraction) < 10:
            fraction.extend(0.0 for _ in range(10 - len(fraction)))
        user, nice, kern, idle, iowait, irq, soft, steal, guser, gnice = fraction
        fgs = [self.get_gradient(v, 0.0, 1.0, lh, hh, ls, hs, lv, hv)
               for v, lh, hh, ls, hs, lv, hv in
               zip(fraction,
                   self.low_hues, self.high_hues,
                   self.low_sats, self.high_sats,
                   self.low_vals, self.high_vals,
               )
        ]

        block = super().run_common(short)
        block['full_text'] = ''.join(
            self.get_vert_bar(v, fg=fg, bg=bg)
            if self.short else
            self.get_bar(v, fg=fg, bg=bg)
            for v, fg, bg in zip(fraction, fgs, self.bgs)
        )
        return block

class MemBarProvider(Provider):
    low_hue = 1.0/3.0
    high_hue = 0.0

    def run_common(self, short = False):
        vmem = psutil.virtual_memory()

        block = super().run_common(short)
        block['full_text'] = self.get_vert_bar(vmem.percent, 0.0, 100.0) if short else self.get_bar(vmem.percent, 0.0, 100.0)
        block['color'] = self.get_gradient(vmem.percent)
        return block

class DDateClockProvider(Provider):
    SH_WEEKDAYS = ['SM', 'BT', 'PN', 'PP', 'SO']
    SH_SEASONS = ['CHA', 'DIS', 'CON', 'BUR', 'AFT']

    def run(self):
        dt = ddate.base.DDate()

        if dt.season is None:
            sn = '???'
        else:
            sn = self.SH_SEASONS[dt.season]

        if dt.day_of_week is None:
            wd = '??'
        else:
            wd = self.SH_WEEKDAYS[dt.day_of_week]

        if dt.day_of_season is None:
            sd = '??'
        else:
            sd = '{:02}'.format(dt.day_of_season)

        block = super().run()
        block.update({
            'full_text': '{}-{}-{} {}{}'.format(
                dt.year, sn, sd, wd,
                '' if dt.holiday is None else (' ' + dt.holiday),
            ),
            'color': '#7700FF' if dt.holiday is None else '#FF00FF',
        })
        return block

    def run_short(self):
        block = super().run_short()
        dt = ddate.base.DDate()
        block.update(
                full_text = 'DIS',
                color = '#7700FF' if dt.holiday is None else '#FF00FF',
        )
        return block

class SimpleClockProvider(Provider):
    format = '%Y-%m-%d<span color="#0000ff">T</span>%H:%M:%S<span color="#555555">.%Nm</span><span color="#007700">%z %Z W%W J%j</span>'
    format_short = '%H:%M:%S<span foreground="#555555">.%Nd</span>'
    subsecs_pattern = re.compile('%N([dcmunf])')

    def run_common(self, short = False):
        block = super().run_common(short)
        # Hypothesis: since its precision is lacking in struct_time, there is no fractional time offset
        now = time.clock_gettime(time.CLOCK_REALTIME)
        frac = now - int(now)
        subs = {
                'd': '{:01d}'.format(int(frac*10)),
                'c': '{:02d}'.format(int(frac*100)),
                'm': '{:03d}'.format(int(frac*1000)),
                'u': '{:06d}'.format(int(frac*1000000)),
                'n': '{:09d}'.format(int(frac*1000000000)),
                'f': str(frac),
        }
        def repl(mo):
            return subs[mo.group(1)]
        fo = self.subsecs_pattern.sub(repl, self.format_short if short else self.format)
        block.update({
            'full_text': time.strftime(fo, getattr(self, 'timefunc', time.localtime)(now)),
            'color': self.color,
            'markup': 'pango',
        })
        return block

class CentClockProvider(Provider):
    format = '{ch:01d}:{cm:02d}:{cs:02d}'
    color = '#007777'

    def run_common(self, short = False):
        tinfo = getattr(self, 'timefunc', time.localtime)()
        secs = tinfo.tm_hour * 3600 + tinfo.tm_min * 60 + tinfo.tm_sec
        ch, cm, cs = secs // 10000, secs // 100 % 100, secs % 100
        block = super().run_common(short)
        block.update({
            'full_text': self.format.format(ch=ch, cm=cm, cs=cs),
            'color': self.color,
            'markup': 'pango',
        })
        return block

class UNIXClockProvider(Provider):
    format = '{ut:.3f}'
    format_short = 'UT'
    clock = time.CLOCK_REALTIME
    color = '#777777'

    def run(self):
        block = super().run()
        block.update(
            full_text = self.format.format(ut = time.clock_gettime(self.clock)),
            color = self.color
        )
        return block

    def run_short(self):
        block = super().run_short()
        block.update(full_text = self.format_short, color = self.color)
        return block

class UTDiffClockProvider(SimpleClockProvider):
    timefunc = time.gmtime
    format = '<error>'
    color = '#777700'

    def run(self):
        lt = time.localtime()
        gt = time.gmtime()
        fmt = ''
        if lt.tm_year != gt.tm_year:
            fmt += '%Y-'
        if lt.tm_mon != gt.tm_mon:
            fmt += '%m-'
        if lt.tm_mday != gt.tm_mday:
            fmt += '%d'
        fmt += '<span color="#0000ff">T</span>%H:%M:%SZ'
        if lt.tm_wday > gt.tm_wday:
            fmt += ' W%W'
        if lt.tm_yday != gt.tm_yday:
            fmt += ' J%j'

        block = super().run()
        block.update({
            'full_text': time.strftime(fmt, gt),
            'color': self.color,
            'markup': 'pango',
        })
        return block

    def run_short(self):
        block = super().run_short()
        block['full_text'] = 'UTC'
        return block

class WaiterInfo(Provider):
    other_format = 'I:{intv:.02f},SB:{sowb:.03f},EB:{eowb:.03f}'
    other_format_short = 'TC'

    sbw_format = 'I:{intv:.02f},dT:{bias:.03f},dI:{ival:.03f},SB:{sowb:.03f},EB:{eowb:.03f}'
    sbw_format_short = 'TC'

    times = (0.01, 0.05, 0.1, 0.5, 1.0)
    time_cur = 4

    def __init__(self, waiter):
        self.waiter = waiter

    def run_common(self, short = False):
        block = super().run_common(short)
        intv = self.waiter.interval
        sowt = getattr(self.waiter, 'start_time', 0.0)
        eowt = getattr(self.waiter, 'end_time', 0.0)
        common = {
                'intv': intv,
                'sowt': sowt,
                'eowt': eowt,
                'sowb': sowt % intv,
                'eowb': eowt % intv,
        }
        if isinstance(self.waiter, SleepBiasWaiter):
            block['full_text'] = (self.sbw_format_short if short else self.sbw_format).format(
                    bias=self.waiter.bias_corr,
                    ival=self.waiter.interval_corr * self.waiter.interval,
                    **common
            )
        else:
            block['full_text'] = (self.other_format_short if short else self.other_format).format(
                    **common
            )
        return block

    def click(self, block):
        if super().click(block):
            return True
        if block['button'] in (4, 5):
            if block['button'] == 4:
                self.time_cur += 1
                if self.time_cur >= len(self.times):
                    self.time_cur = len(self.times) - 1
            else:
                self.time_cur -= 1
                if self.time_cur < 0:
                    self.time_cur = 0
            self.waiter.interval = self.times[self.time_cur]
            self.waiter.reset = True
            return True
        return False

if __name__ == '__main__':
    dp_root = DiskProvider('/')
    dp_root.color = '#0077ff'
    #dp_home = DiskProvider('/home')
    #dp_home.color = '#00ffff'
    la = LoadAverageProvider()
    la.color = '#000077'
    bp = BatteryProvider()
    bp.voltage_high = 8.45
    # A little bias to keep the bar clock ticking at a consistent rate
    #waiter = SleepBiasWaiter(1.0)
    waiter = PosixTimerWaiter(1.0)
    bsi = WaiterInfo(waiter)
    bsi.color = '#770077'
    #bsi.short = False
    sc = SimpleClockProvider()
    sc.short = False
    sc.priority = 10
    ut1 = UNIXClockProvider()
    ut2 = UNIXClockProvider()
    ut2.clock = time.CLOCK_MONOTONIC
    ut2.color = '#000077'
    ut2.format_short = 'BT'
    st = Status(
        dp_root,
        #dp_home,
        NetworkProvider(),
        *TemperatureProvider.all(),
        bp,
        la,
        CPUBarProvider(),
        MemBarProvider(),
        ut2,
        ut1,
        CentClockProvider(),
        UTDiffClockProvider(),
        sc,
        DDateClockProvider(),
        bsi,
        waiter = waiter,
    )
    for prov in st.providers:
        if not isinstance(prov, SimpleClockProvider):
            prov.interval = 0.5
    st.run()
