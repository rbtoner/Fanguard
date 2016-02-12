# FanGuard: The Smart Tumblr Spoiler Filter
Backup of modules and macros for my Insight Data Science SV 2016A project.
* Resulting app: [http://fanguard.xyz/](http://fanguard.xyz/)
* Explanation of algorithm: [slides](https://docs.google.com/presentation/d/1L5Ib0TfZW-So6c2YW9oTKqWZrr1DMfvrf_NA6PVlbIQ/edit?pref=2&pli=1#slide=id.g1093b050e1_2_133)

## Contents of repository:
### AlgorithmTrain: Algorithm training functions and associated algorithms
* ***AnaFunc.py***: High-level and drawing functions
* ***dfmaker.py***: Functions for constructing dataframes, modifying cleaned data, etc.
* ***modelmaker.py***: Vocabulary and model creation, CV and training lower-level functions

### DataCollection: Data collection and cleaning (from Tumblr API)
* ***postGather.py***: Functions to gather posts from the Tumblr API (employs [pytumblr](https://github.com/tumblr/pytumblr))
* ***cleaners.py***: Cleaning functions

### macros: Collection of ipython notebooks for analysis
* ***AuthorTest.ipynb***: Testing effects of repeat authorship on model
* ***DataGather.ipynb***: Gathering posts from 9 separate content sources, cleaning, and storing to MySQL database.
* ***Model_SpoilerFilter_CV.ipynb***: Cross-validation of models and optimization of parameters.
* ***Model_Train_PreFilter.ipynb***: Training for Pre-Filter vocabulary and check for accuracy.
* ***Model_Train_SpoilerFilter.ipynb***: Training of Spoiler Filter, check for accuracy (ROC), with diagnostic plots.
