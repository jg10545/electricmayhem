# whitebox physical patch pipeline

## Usage

### To train a patch

* Initialize all the steps of your pipeline
* Combine into a `Pipeline` object
* Call `pipeline.set_logging()` to point to a log directory and MLFlow server
* Define a loss function and call `pipeline.set_loss()` to add it to the pipeline. 
  * The loss function should have two inputs: the output of the pipeline, and the patch parameters
  * The loss function should output a dictionary mapping loss function terms to UNREDUCED values. So each value in the dictionary should be a tensor of length `batch_size`
  * `pipeline.set_loss()` will run a random patch through to test the loss function; you may need to manually specify the shape.
* Call `pipeline.initialize_patch_params()` to create a new patch (or latent vector for a patch)
* Call `pipeline.cuda()` to copy to GPU
* Call `pipeline.train_patch()` to train the patch. Main hyperparameters you'll need to specify:
  * batch size
  * number of steps
  * loss function weights- you'll need to pass at least one of the keys from the dictionary your loss function outputs (summary stats on all the loss dict outputs will still be stored in TensorBoard though)



### To optimize hyperparameters

* Initialize all the steps of your pipeline
* Combine into a `Pipeline` object
* Define a loss function and call `pipeline.set_loss()` to add it to the pipeline
* Call `pipeline.cuda()` to copy to GPU
* Call `pipeline.optimize()` to start training patches with randomly-selected hyperparameters. Specify hyperparameters the same way you would for `pipeline.train_patch()`, except:
  * replace the hyperparameter value with a tuple `(low,high)` to optimize that value (on a linear scale)
  * replace with a tuple `(low, high, 'log')` to optimize on a log scale
  * replace with a tuple `(low, high, 'int')` to optimize as an integer (for example, for the number of gradient accumulation steps)
  * replace with a list of strings `['option1', 'option2']` to optimize categorical hyperparameters like the optimizer (`'adam'`, `'bim'`, or `'mifgsm'`) or learning rate decay schedule (`'none'`, `'cosine'`, `'exponential'`, or `'plateau'`) 



## General pipeline stages

* **create:** create a patch from some parameterization of the patch
* **implant:** incorporate the patch into a target image
* **compose:** create the final image that will be passed through the target model
* **infer:** run the composed image through one or more models
* **loss:** compute loss and performance metrics on the model result; backpropagate gradients.

## Steps implemented so far

### create

* `PatchSaver`: pass-through to make sure your patch gets logged to TensorBard
* `PatchResizer`: upsample a low-res patch
* `PatchTiler`: create a large patch from a small one by tiling it
* `PatchStacker`: stack a 1-channel patch into a 3-channel patch
* `PatchScroller`: translate the patch with toroidal boundary conditions during training; makes some patch objectives less like an inversion attack
* `HighPassFilter`: run the patch through a 2D high-pass filter during training; makes some patch objectives less like an inversion attack
* `SpectrumSimulationAttack`: during training steps, applies the domain noise from "Frequency Domain Model Augmentation for Adversarial Attack" by Long et al (2022)
* `SoftProofer`: on evaluation steps only, use a color management system and ICC profile to simulate what the patch will look like after printing

### implant

* `RectanglePatchImplanter`: implant a patch into a target image. Can randomly select from multiple images and multiple bounding boxes per image. Optionally, reserve some target images only for evaluation.
* `FixedRatioRectanglePatchImplanter`: scale patch to a fixed size with respect to the target box and implant in image. Optionally, reserve some target images only for evaluation.
* `ScaleToBoxRectanglePatchImplanter`: scale the patch to fill the box.
* `WarpPatchImplanter`: input arbitrary corner coordinates instead of rectangular boxes; deforms patch to meet corners.

### compose

* `KorniaAugmentationPipeline`: wraps the `kornia` library to augment an image

### infer

* `ModelWrapper`: wrap a pytorch model as a pipeline step
* `YOLOWrapper`: model wrapper that adds a detection visualization to TensorBoard

### Loss functions in `electricmayhem.whitebox.loss`

* `NPSCalculator` for non-printability score
* `saliency_loss` alternative to NPS
* `total_variation_loss` to penalize high-frequency components in patch

## Extending

Every pipeline step should subclass `electricmayhem._pipeline.PipelineBase`, which in turn subclasses `toch.nn.Module`. Make sure:

### Required steps

* There should be an `__init__()` method that calls `super().__init__()`. 
* Any keyword arguments you need to re-initialize the step should be captured in a JSON/YAML-serializable dict in `self.params`.
* There should be a `forward()` method that does a few things:
  * If called with `control=True`, runs a control batch (same configuration as previous batch but without the patch)
  * If called with `evaluate=True`, runs an evaluation batch (for example possibly using holdout images or a separate model)
  * If a dictionary of paramaters is passed to the `params` kwarg, overrules any randomly-sampled parameters with these values.
  * Can input `**kwargs`
  * Returns a 2-tuple containing that stages' output and the input `kwargs` dictionary (possible with more information added to it)
* There should be a `get_last_sample_as_dict()` method. It should return any stochastic parameters sampled for the last batch as a dictionary containing lists or 1D `numpy` arrays of length `batchsize`

### Optional steps

* Overwrite the `get_description()` method to generate a more useful markdown description for MLFlow.
* Overwrite the `log_vizualizations()` method with any diagnostics that would be useful to log to TensorBoard. This method will get called whenever `pipeline.evaluate()` is run.
* Overwrite the `validate()` method to check for anything specific that could go wrong with that step. When the user calls `Pipeline.validate()` it will run the `validate()` method for each step. Use the `logging` library to record check results at the `info` or `warning` level.

```
class MyPipelineStage(PipelineBase):

    def __init__(self, foo, bar):
        super().__init__()
        self.params = {"foo":foo, "bar":bar}
        
    def forward(self, x, control=False, evaluate=False, params=None, **kwargs):
        <stuff here>
        y = f(x)
        return y, kwargs
        
    def get_last_sample_as_dict(self):
        return dict(<some stuff>)
        
    def log_vizualizations(self, x, x_control, writer, step, logging_to_mlflow=False):
        """
        """
        writer.add_image("stacked_patch", <some stuff>, global_step=step,
                        logging_to_mlflow)
         
    def get_description(self):
        return "**MyPipelineStage** and some details that would be helpful in mlflow"
        

```