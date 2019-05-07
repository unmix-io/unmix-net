# unmix-net

Tensorflow based neuronal network framework to extract vocals and instrumental from music.
Python 3.7 was used for implementation.

## Dependencies

### Installation

Install all dependencies by using `pip install -r requirements.txt`.

- keras
- tensorflow
- argparse
- Pillow
- matplotlib
- pydot
- numpy
- psutil
- commentjson
- gitpython
- colorama
- progressbar2
- mir_eval
- pytube
- librosa

We highly recommend running the solution on [tensorflow-gpu](https://www.tensorflow.org/install/gpu).

### Docker

Instead of installing the dependencies locally, our [docker image](https://hub.docker.com/r/unmix/unmix) can be used: `docker pull unmix/unmix`.

## Settings

### Configurations

Training runs must be configured with a jsonc configuration file.
Configuration files can inherit from parent configurations:

```json
{
    "base": "default-hourglass.jsonc",
    ...
```

If a property is specified multiple times, the child configurations always overrides.

Every configuration inherits by default from the [master.jsonc](https://github.com/unmix-io/unmix-net/blob/master/configurations/master.jsonc) configuration.
Comments are allowed in jsonc files.

### Environment Variables

Configuration files support access to environment varialbes.

- `UNMIX_CONFIGURATION_FILE`: Path to the configuration file (default parameter for training)
- `UNMIX_COLLECTION_DIR`: Path to the training, validation and test data collection
- `UNMIX_SAMPLE_RATE`: Sample rate of the training data (used for training and prediction)
- `UNMIX_SONG_LIMIT`: Limit amount of songs to be included in the training (for smaller training runs)
- `UNMIX_TEST_FREQUENCY`: Frequency in epochs to run an accuracy test
- `UNMIX_TEST_DATA_COUNT`: Number of songs to include in to the accuracy test
- `UNMIX_LIMIT_ITEMS_PER_SONG`: Limit of batchitems used per song for training

The variables can be added to your operating system or by adding a `.env` file to the (repository) base directory.

## Training

Example call: `python3 train.py --configuration configuration/final/hourglass.jsonc`

### Parameters

- `configuration`: Path to a valid jsonc configuration file. If not specified the value of the `UNMIX_CONFIGURATION_FILE` environment variable is used.
- `workingdir`: Optional working directory where the runs ordner is published

### Procedure

Following a rough overview what happens during a training session:

1. Configuration initialization, output run folder creation
2. Data loading, splitting training, validation and test data
3. Training per epoch with batch generators
4. Write callbacks (logs, weights, ...)
5. Optional: Calculate accuracies
6. Stop if early stopping or epoch count finished

### Output

Every training run generates a "run folder" with the following structure:

- _plots_: Output folder for plots (can be configured otherwise)
- _predictions_: Output folder for predictions and accuracy tests.
- _weights_: Output folder for the trained weights
- _accuracy_x.csv_: [mir_eval](https://craffel.github.io/mir_eval/) based accuracies of the track prediction.
- _results.csv_: Result file including loss, mean prediction, validation loss, validation mean prediction per epoch
- _configuration.jsonc_: Merged configuration file which is used by the training run
- _environment.json_: Environment information including working directories, git version, environment variables
- _logs.txt_: Logfile
- _model.h5_: Model and weights (created after training is finished)
- _model.json_: Model configuration (created after training is finished)

## Prediction

Example call run folder: `python3 predict.py --run_folder runs/20190506-154117-hourglass --song skyfall.mp3`

Example call weights file: `python3 predict.py --weights weights.h5 --configuration configuration.jsonc --song skyfall.mp3`

### Parameters

- `run_folder`: Run folder from a training run (other parameters get derived)
- `configuration`: (Not necessary if `run_folder`) Path to the configuration file
- `weights`: (Not necessary if `run_folder`) Path to a weights file
- `workingdir`: (Not necessary if `run_folder`) Optional working directory
- `sample_rate`: (Not necessary if `run_folder`) Sample rate which was used for training and will be used for prediction
- `song`: Path to a single song to predict (extract vocals and instrumental)
- `songs`: Path to a folder of songs to predict (extract vocals and instrumental)
- `youtube`: Link to a [YouTube](https://www.youtube.com/watch?v=dQw4w9WgXcQ) link to be predicted (extract vocals and instrumental)

### Output

The predicted songs will be written into the working directory.

If a run folder was specified all results are stored in the _predictions_  folder.