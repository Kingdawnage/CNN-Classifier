This is an image classifier made using [tensorflow](https://www.tensorflow.org/) and [python](https://www.python.org/).


## Getting Started
You need to have python installed and any IDE of your choice. I recommend Visual Studio or Visual Studio Code.
If you do not have [python](https://www.python.org/), you can install it here.

## Installing Packages
The `requirements.txt` file contains all the packages needed for the project. Run the commands on the terminal to proceed
```
pip install -r requirements.txt # Each time there is a new package
pip install <package> # Each time you want to install a package
```

## Creating a Virtual Environment
A virtual environment (venv/virtualenv) allows you to manage package installations for different projects.

Make sure you have the latest version of pip by running:

### Unix/macOS:
```
python3 -m pip install --user --upgrade pip
python3 -m pip install --user --upgrade pip
```

### Windows:
```
py -m pip install --upgrade pip
py -m pip --version
```

You can then run the following command to create the venv:

### Unix/macOS:
```
python3 -m venv .venv
```

### Windows:
```
py -m venv .venv
```

Activate the virtual environment with the following:

### Unix/macOS
```
source /.venv/bin/activate
```

### Windows
```
.\.venv\Scripts\activate
```

Finally you can deactivate the virtual environment after you are done
```
deactivate
```

## How to Use

If you followed the setup accordingly, you should have an virtual python environment activated with the required dependencies installed
If that is the case, the program can be run accordingly
```
py ./src/source.py
```

### Train the model
To train a new model, uncomment `lines 11 to 94`. By default it uses a pretrained model to make predictions. 
If you would like to save your trained model you can uncomment `line 89`

### Making predictions
By default, predictions are made based on the content of the prediction directory
```
./assets/prediction
```
Predictions across more objects can be done by just adding the images of choice the directory.
Predictions using the pretrained model or your trained model can be done rather simply. Refer to `lines 112 to 116`

## Expansion

The project can be expanded to include multiple classes. Right now it can only make predictions across two classes. To add more classes
just add training and validation data of the concerned class to their respective directories

```
./assets/training/"concerned class"
./assets/validation/"concerned class"
```

After the model has been retrained, add a picture of the concerned class to the predictions directory and view your results
