import json, sys, time, os, colorsys, re

import netifaces, psutil, ddate.base

class Status(object):
    def __init__(self, *providers, interval = 0.5):
        self.providers = list(providers)
        self.interval = interval
        self.stop = True

    def run(self):
        json.dump({'version': 1}, sys.stdout)
        print('\n[')
        self.stop = False
        while not self.stop:
            op = [i.process() for i in self.providers]
            json.dump(op, sys.stdout)
            print(',', flush=True)
            time.sleep(self.interval)

class Provider(object):
    color = '#ffffff'
    cached = {}

    def run(self):
        return {'color': self.color}

    interval = None
    last_run = None
    def should_run(self):
        if self.interval is None:
            return True
        if self.last_run is not None and self.last_run + self.interval > time.time():
            return False
        self.last_run = time.time()
        return True

    def process(self):
        if self.should_run():
            try:
                self.cached = self.run()
            except Exception as e:
                self.cached = self.err_block(e)
        return self.cached

    def err_block(self, exc):
        return {'color': '#ff00ff', 'full_text': str(exc)}

    low_hue = 0.0
    high_hue = 2.0/3.0
    low_value = 0.0
    high_value = 100.0

    def get_gradient(self, v, sat=1.0, val=1.0, l=None, h=None, lh=None, hh=None):
        if l is None:
            l = self.low_value
        if h is None:
            h = self.high_value
        if lh is None:
            lh = self.low_hue
        if hh is None:
            hh = self.high_hue
        clamped = min((h, max((l, v))))
        r, g, b = colorsys.hsv_to_rgb(
            lh + (hh - lh) * (clamped - l) / (h - l),
            sat, val,
        )
        r, g, b = tuple(min((int(i*256), 255)) for i in (r, g, b))
        return '#{:02x}{:02x}{:02x}'.format(r, g, b)

    BARS = [' ', '▏', '▎', '▍', '▌', '▋', '▋', '▊', '▊', '█']
    BLOCK = '█' 

    def get_bar(self, v, l=0.0, h=1.0):
        p = int(100.0 * (v - l) / (h - l))
        t, o = divmod(p, 10)
        if o == 0:
            return (self.BLOCK * t).ljust(10)
        return (self.BLOCK * t + self.BARS[o]).ljust(10)

    VERTS = ' _▁▂▃▄▅▆▇█'

    def get_vert_bar(self, v, l=0.0, h=1.0):
        n = (v - l) / (h - l)
        return self.VERTS[min((len(self.VERTS) - 1, max((0, int(n / len(self.VERTS))))))]

class DiskProvider(Provider):
    crit_free = 1024**3
    format = '{free:.2f}G'
    div = 1024**3

    def __init__(self, path):
        self.path = path

    def run(self):
        vfs = os.statvfs(self.path)

        free = vfs.f_bfree * vfs.f_bsize

        fd = {
            'total': vfs.f_blocks * vfs.f_bsize / self.div,
            'free': free / self.div,
        }

        block = super().run()
        block['full_text'] = self.format.format(**fd)
        block['urgent'] = free < self.crit_free
        return block

class NetworkProvider(Provider):
    format_down = '!{interface}'
    color_down = '#333333'
    format_up = '{addr}/{bits}'
    color_up = '#777777'

    hide = re.compile(r'lo|sit.*|ip6tnl.*|.*\.\d+')
    priority = (netifaces.AF_INET, netifaces.AF_INET6)
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
            for af in self.priority:
                data = addrs.get(af)
                if not data:
                    continue
                data = data[0]
                data['interface'] = iface
                data['bits'] = self.bits_in_ipv4(data['netmask']) if af == netifaces.AF_INET else int(self.ENDMASK.match(data['netmask']).group(1))
                texts.append('<span foreground="{}">{}</span>'.format(self.color_up, self.format_up.format(**data)))
                break
            else:
                texts.append('<span foreground="{}">{}</span>'.format(self.color_down, self.format_down.format(interface=iface)))

        block = super().run()
        block['full_text'] = self.sep.join(texts)
        block['markup'] = 'pango'
        return block

class TemperatureProvider(Provider):
    low_value = 20.0
    low_hue = 2.0/3.0
    high_value = 70.0
    high_hue = 0.0
    crit_temp = 80.0
    format = '{temp:.0f}C'

    def __init__(self, path='/sys/class/thermal/thermal_zone0/temp'):
        self.path = path
        self.f = open(self.path)

    def run(self):
        self.f.seek(0)
        temp = float(self.f.read()) / 1000.0

        block = super().run()
        block['full_text'] = self.format.format(temp=temp)
        block['color'] = self.get_gradient(temp)
        block['urgent'] = temp >= self.crit_temp
        return block

class BatteryProvider(Provider):
    format = '{status} {remaining} {flow} {voltage} {percentage:.2f}% {bar}'

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
        block = super().run()
        block['full_text'] = '{battery} MISSING'.format(battery=self.battery)
        block['color'] = self.missing_color
        block['urgent'] = self.missing_urgent
        return block

    def run(self):
        if self.f is None:
            try:
                self.f = open('/sys/class/power_supply/{}/uevent'.format(self.battery))
            except OSError:
                return self.block_missing()

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
        info['percentage'] = 100.0 * partial

        block = super().run()
        block['full_text'] = self.format.format(**info)
        block['color'] = self.get_gradient(partial)
        block['urgent'] = partial <= self.crit_value and info['STATUS'] == 'Discharging'
        block['markup'] = 'pango'
        return block

class LoadAverageProvider(Provider):
    path = '/proc/loadavg'
    format = '{avg1} {avg5}'
    crit_load = 8.0

    def __init__(self, path=None):
        if path is not None:
            self.path = path
        self.f = open(self.path)

    def run(self):
        self.f.seek(0)
        avg1, avg5, avg15, tasks, lastpid = self.f.read().split(' ', 5)

        block = super().run()
        block['full_text'] = self.format.format(avg1=avg1, avg5=avg5, avg15=avg15, tasks=tasks)
        block['urgent'] = float(avg1) >= self.crit_load
        return block

class CPUBarProvider(Provider):
    path = '/proc/stat'

    low_value = 0.0
    high_value = 1.0
    low_hue = 2.0/3.0
    high_hue = 5.0/6.0

    def __init__(self):
        self.last_total = 0
        self.last_busy = 0
        self.f = open(self.path)

    def run(self):
        self.f.seek(0)
        parts = list(map(int, self.f.readline().strip().split()[1:]))

        total = sum(parts)
        del parts[3:5]
        busy = sum(parts)

        value = (busy - self.last_busy) / (total - self.last_total)
        
        self.last_total = total
        self.last_busy = busy

        block = super().run()
        block['full_text'] = self.get_bar(value)
        block['color'] = self.get_gradient(value)
        return block

class MemBarProvider(Provider):
    low_hue = 1.0/3.0
    high_hue = 0.0

    def run(self):
        vmem = psutil.virtual_memory()

        block = super().run()
        block['full_text'] = self.get_bar(vmem.percent, 0.0, 100.0)
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

class SimpleClockProvider(Provider):
    format = '%Y-%m-%d<span foreground="#0000ff">T</span>%H:%M:%S<span color="#007700">%z %Z W%W J%j</span>'

    def run(self):
        block = super().run()
        block.update({
            'full_text': time.strftime(self.format, getattr(self, 'timefunc', time.localtime)()),
            'color': self.color,
            'markup': 'pango',
        })
        return block

class GMDiffClockProvider(SimpleClockProvider):
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

if __name__ == '__main__':
    dp_root = DiskProvider('/')
    dp_root.color = '#0077ff'
    dp_home = DiskProvider('/home')
    dp_home.color = '#00ffff'
    la = LoadAverageProvider()
    la.color = '#000077'
    Status(
        dp_root,
        dp_home,
        NetworkProvider(),
        BatteryProvider(),
        TemperatureProvider(),
        la,
        CPUBarProvider(),
        MemBarProvider(),
        GMDiffClockProvider(),
        SimpleClockProvider(),
        DDateClockProvider(),
    ).run()
