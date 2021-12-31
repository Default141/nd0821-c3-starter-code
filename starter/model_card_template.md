# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
    Model use for predict salary
## Intended Use
    Use for predict salary with api interface
## Training Data
    Processed data from census.csv split to train and test and 
    process encode with function process_data()
## Evaluation Data
    Processed data from census.csv split to train and test and 
    process encode with function process_data()
## Metrics
    precision    recall     fbeta
     0.884713  0.462383  0.607346
## Ethical Considerations
    Model might has bias with education data might due to lacking of data 
## Caveats and Recommendations
    This model are still in exerimant process result from this
    model are not alway certain accurate
