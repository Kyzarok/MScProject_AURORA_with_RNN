# MScProjectRepo
Extending Autonomous Skill Discovery with Recurrent Neural Networks

This work is based off of the original research done by Dr Antoine Cully from "Autonomous skill discovery with Quality-Diversity and Unsupervised Descriptors" ( https://arxiv.org/abs/1905.11874)

This repo contains the original code (OriginalCode) for the ballistic task as well as my implementation (BallisticMyVer).

Running the code in BallisticMyVer requires the following dependencies:
* Tensorflow version 1
* numpy
* datetime
* matplotlib
* random
* math

You can run this code with the command "**$python3 control.py**".
This command line can take multiple arguments:
* --**version** : A string that defines which algorithm to run. The default is "null". Can take as argument:
  * "**GT**" to generate a reference ground truth distribution 
  * "**handcoded**" to run the handcoded method
  * "**genotype**" to run the genotype method
  * "**pretrained**" to run the pretrained version of AURORA-AE
  * "**incremental**" to run the incremental version of AURORA-AE
* --**with_RNN** : A boolean that sets whether or not to run the code with an LSTM layer on the AE versions. The default is "False"
* --**plot_runs** : An int that will choose which version to plot. Choose depending on what data you have. If you have run the handcoded, genotype, standard pretrained and standard incremental versions and not run any LSTM version, enter "1". If you have run all of the variations, enter "2". The default is "0".
* --**num_epochs** : An int that sets the number of epochs that will be used to train the AEs in the AE versions. It is recommended to set this to 100 when running the LSTM variants unless you have a powerful graphics card. The default is "500".
* --**everything** : A boolean that, if True, will run everything in the project and output the plots at the very end. The "I'm lazy" option. WARNING: This sometimes glitches out after long runs for some reason so honestly I'd recommend doing eachg variation individually. The default is "False".

For example if you wanted to run the handcoded version, input "**$python3 control.py --version handcoded**"
