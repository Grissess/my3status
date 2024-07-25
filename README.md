my3status
=========

This is a personal-use, Python3-driven status bar that I use on all my i3/sway
devices. I'm not sure if it'll help you, general user, but you're more than
welcome to take it.

Here's a little screenshot on my desktop:

![scrot](../eg/scrot.png)

And on my laptop, with most everything collapsed:

![scrotshort](../eg/scrotshort.png)

My philosophy is about colors more than numbers; I like being able to tell how
much battery is left on my laptop from across the room by the color of the
capacity bar. Similarly, the red/magenta pairings for physmem/CPU indicate that
something is happening, or requires my attention. (This is my response to the
now-redacted claim that memory monitoring on i3status is unnecessary, by the
way: I run swapless systems, where physmem _fullness_ is an important metric
for me. The color makes it easy to judge this at a glance.)

This script is extensible, but hardly well-packaged. If you're looking for
something more standardized, consider
[i3pystatus](https://pypi.org/project/i3pystatus/), the qualms with which
encouraged me to write my own streamlined version, or [alexbakker's project of
the same name](https://github.com/alexbakker/my3status), which I discovered
afterward, and seems both functionally equivalent and more professional than
this :)

Installation
------------

As it's already gitignored and shebanged, the best place to put a Python Venv
is in `venv`. Thus, from this repository:

```
python3 -m venv venv
. venv/bin/activate
pip3 install -r requirements.txt
```

Then you can `./status.py` from within the same directory.

If you don't have control over the directory, the somewhat longer invocation

```
$REPO/venv/bin/python $REPO/status.py
```

will also suffice, provided `$REPO` refers to this repository root.

Usage
-----

Pop this into a place you know about, and set it off in your config with:

	bar {
		status_command python3 /path/to/status.py
		# And any other config...
	}

This works with both Sway and i3, to my knowledge, since both conveniently use
the same streaming JSON IPC protocol.

As of the version which is being here documented, the modules support a "short"
mode, which makes them more compact on especially narrow displays (or when you
have a lot of workspaces open). Modules start "short" by default, and can be
left-clicked to toggle. Modules also can handle clicks in other ways, but none
yet do.

Changing what displays, or the order, is done with your favorite text editor.
Just change the `Provider` instances passed to the `Status` instance. Feel free
to use the rest of the implementation as reference.

Dependencies
------------

Presently, this wants these packages:

- `netifaces`, for network interface information (addresses, mostly)
- `psutil`, for memory info
- `ddate`, for the Discordian calendar

If you remove the relevant providers, you can, of course, edit this list down.
Presently, that also entails editing the imports on line 3.

Contributions
-------------

I use this personally, so don't expect me to put the kitchen sink in here, or
to be very generous with my inclusion of features _I_ find unnecessary.
However, you're certainly free to fork this and modify it on your own under the
terms of the GPLv3, and I could be convinced to merge a PR if it fixes a bug or
adds a feature I personally like.
