# NED_Asteroid_Energy_Deposition

This script is used to initialize a hydrocode simulation of an asteroid deflection/disruption with a nuclear explosive device for planetary defense purposes. It generates an angle/depth-dependent internal energy profile on the surface of the asteroid, which mimics the radiation absorbed from a nuclear explosive device. Please see the the Planetary Science Journal article "X-Ray Energy Deposition Model for Simulating Asteroid Response to a Nuclear Planetary Defense Mitigation Mission" for more information and details on implementation. 

## Getting Started

Clone the git repository to a convenient location. You can copy the ``energyDep.py`` module to your Python site-packages directory if you want it globally accessible.

### Prerequisites

The Python scripts load [NumPy](https://numpy.org) and [SciPy](https://www.scipy.org).

## The deposition function

The script ``energyDep.py`` provides the function ``Edepfunc(z, Mat, BB, cosang, Flx, Tsrc, Porosity)``

The inputs are:
| argument | description   |
| :--------| :------------ |
| z        | depth from surface in cm |
| Mat      | one of ['SiO2', 'Forsterite', 'Ice', 'Iron'] |
| BB       | source temperature, one of [1.0, 2.0] |
| cosang   | cosine of angle of incidence  |
| Flx      | fluence in kt/m^2  |
| Tsrc     | length of source in ns |
| Porosity | porosity in range (0.0, 1.0)|

The output is the energy density in Perg/cm^3

Note that for generality the function depends on the angle of incidence. 
For a spherical asteroid this can be calculated from the angle measured from the asteroid's center, the radius of the asteroid, and the height of burst.

## Test script

The script ``CalcEdep.py`` integrates the energy deposited in the asteroid and compares it to the energy incident on the asteroid. 

```
python CalcEdep.py
```
## EOS files

The two LEOS  and the Quartz equations of state are given in a Sesame format.
- Ice_L2016_sesame.txt
- Iron_L261_sesame.txt
- Quartz_L4358_sesame.txt
There are comments with a '#' in the first column that label each table
In addition the standard Sesame record lines and any comment records are also preceded by a '#'

The files contain these tables:
- the cold curve, Pc and Ec
- the total EOS, PT, Et, Cs^2, St
- the effective charge state, Zeff
- the melt temperature, Tm

For the 2D tables density varies most rapidly in the table

The Quartz EOS is also available at (https://github.com/isale-code/M-ANEOS). The quartz input file is in the M-ANEOS/input directory. 

The Forsterite EOS is available at (https://github.com/ststewart/aneos-forsterite-2019). The Sesame text files are included.

Units:
- density, g/cm^3 = Mg/m^3
- temperature, K
- pressure, GPa
- specific energy, MJ/kg
- sound speed squared, (km/s)^2
- entropy, MJ/(kg K)

For cases where accessing the melt information from the EOS is difficult we also provide ``Tmelt.py`` with numpy arrays of the melt temperatures from each of the four EOSes given above.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Code of Conduct

Our code of conduct is available at [Code of Conduct](CODE_OF_CONDUCT.md).

## Authors

- **Mary Burkey**  (https://people.llnl.gov/burkey1)
- **Robert Managan**  (https://people.llnl.gov/managan1)

See also the list of [contributors](CONTRIBUTING.md) who participated in this project.

## License

This project is licensed under the BSD-3 License - see the [LICENSE.md](LICENSE.md) file for details

Unlimited Open Source - BSD 3-clause Distribution LLNL-CODE-853602

## SPDX usage

Individual files contain SPDX tags instead of the full license text.
This enables machine processing of license information based on the SPDX
License Identifiers that are available here: https://spdx.org/licenses/

Files that are licensed as BSD 3-Clause contain the following
text in the license header:

    SPDX-License-Identifier: (BSD-3-Clause)



