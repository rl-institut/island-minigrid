# Island minigrid model
This repository contains an oemof.solph-based energy system model designed to simulate minigrids for island systems, 
with a focus on integrating hydrogen (H₂) components. It also includes the option to distinguish between critical demand,
which is always fulfilled, and non-critical demand, which will be fulfilled only if the excess energy allows for it. 

The model allows for flexible configuration of the input parameters via an excel file. To configure the system, please 
adapt the included `input_case.xlsx` file as required. 

After running the simulation, a dashboard containing graphics and tables to explore simulation results will be available
under http://localhost:8060.

## Get started
Clone the repository with `git clone` and create a virtual environment with your favorite environment manager.

Install the requirements

```bash
    pip install -r requirements.txt
```

Run the model using your custom input file

```bash
    python run_simulation.py -i your_input_case_filename.xlsx
```

## Contributing
Please use `black .` before you commit to lint the code