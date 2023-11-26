[![License][s1]][li]

[s1]: https://img.shields.io/badge/license-MIT-blue.svg

[li]: https://raw.githubusercontent.com/matt77hias/cu-smallpt/master/LICENSE.txt

# cpp-smallpt

## About
CUDA C++ modification of Kevin Baeson's [99 line C++ path tracer](http://www.kevinbeason.com/smallpt/)

**Note**: I deliberately chose for the same software design for [all programming languages](https://github.com/matt77hias/smallpt) out of clarity and performance reasons (this can conflict with the nature of declarative/functional programming languages).

<p align="center"><img src="https://github.com/matt77hias/smallpt/blob/master/res/image.png" ></p>

## Before Use
**Modify Windows Timeout Detection and Recovery**

1. Open *regedit*
2. Goto <code>HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers</code>
3. Add (or update) <code>REG_DWORD</code> with name <code>TdrDelay</code> and value <code>x</code> seconds (default is 2 seconds)
4. Reboot Windows

*or* 

**Disable Windows Timeout Detection and Recovery**

1. Open *NVidia's Nsight Monitor*
2. Goto Nsight Monitor options -> General
3. Set <code>WDDM TDR enabled</code> to <code>false</code>
4. Reboot Windows

Also possible via *regedit*
