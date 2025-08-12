from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import IsotonicRegression
from pyspark.sql import DataFrame
from pyspark.sql.functions import when, col

def fit_isotonic_calibrator(valid_df: DataFrame,
                            score_col: str = "prob",
                            label_col: str = "label",
                            output_col: str = "calibrated_prob") -> PipelineModel:
    calib_df = valid_df.withColumn("binary_label", col(label_col))
    
    assembler = VectorAssembler(
        inputCols=[score_col],
        outputCol="calib_features"
    )
    
    isotonic = IsotonicRegression(
        featuresCol="calib_features",
        labelCol="binary_label",
        predictionCol=output_col,
        isotonic=True
    )
    
    pipeline = Pipeline(stages=[assembler, isotonic])
    
    return pipeline.fit(calib_df)

def apply_isotonic(df: DataFrame,
                   calibrator: PipelineModel) -> DataFrame:
    return calibrator.transform(df)
