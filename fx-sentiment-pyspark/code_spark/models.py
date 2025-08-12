from typing import List, Optional, Dict
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler, SQLTransformer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, count

def build_preprocess_stages(feature_cols: List[str],
                            with_scaler: bool = True) -> List:
    stages = []
    
    imputer = Imputer(
        inputCols=feature_cols,
        outputCols=[f"{col}_imputed" for col in feature_cols],
        strategy="median"
    )
    stages.append(imputer)
    
    assembler = VectorAssembler(
        inputCols=[f"{col}_imputed" for col in feature_cols],
        outputCol="features_raw"
    )
    stages.append(assembler)
    
    if with_scaler:
        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withMean=True,
            withStd=True
        )
        stages.append(scaler)
    else:
        renamer = SQLTransformer(
            statement="SELECT *, features_raw as features FROM __THIS__"
        )
        stages.append(renamer)
    
    return stages

def compute_class_weights(train_df: DataFrame,
                          label_col: str = "label",
                          weight_col: str = "weight") -> DataFrame:
    class_counts = train_df.groupBy(label_col).count().collect()
    total_count = sum(row["count"] for row in class_counts)
    
    weight_map = {}
    for row in class_counts:
        label_val = row[label_col]
        class_count = row["count"]
        weight = total_count / (len(class_counts) * class_count)
        weight_map[label_val] = weight
        print(f"Class {label_val}: {class_count} samples, weight = {weight:.3f}")
    
    weight_expr = col(label_col)
    for label_val, weight in weight_map.items():
        weight_expr = when(col(label_col) == label_val, weight).otherwise(weight_expr)
    
    return train_df.withColumn(weight_col, weight_expr)

def lr_pipeline(feature_cols: List[str],
                elastic_net: float = 0.0,
                max_iter: int = 100,
                weight_col: Optional[str] = "weight") -> Pipeline:
    stages = build_preprocess_stages(feature_cols)
    
    lr_params = {
        "featuresCol": "features",
        "labelCol": "label",
        "predictionCol": "prediction",
        "probabilityCol": "probability",
        "rawPredictionCol": "rawPrediction",
        "elasticNetParam": elastic_net,
        "maxIter": max_iter
    }
    
    if weight_col:
        lr_params["weightCol"] = weight_col
    
    lr = LogisticRegression(**lr_params)
    stages.append(lr)
    
    return Pipeline(stages=stages)

def rf_pipeline(feature_cols: List[str],
                num_trees: int = 200,
                max_depth: int = 8) -> Pipeline:
    stages = build_preprocess_stages(feature_cols)
    
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        probabilityCol="probability",
        rawPredictionCol="rawPrediction",
        numTrees=num_trees,
        maxDepth=max_depth,
        seed=42
    )
    stages.append(rf)
    
    return Pipeline(stages=stages)

def gbt_pipeline(feature_cols: List[str],
                 max_iter: int = 200,
                 max_depth: int = 5) -> Pipeline:
    stages = build_preprocess_stages(feature_cols)
    
    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        maxIter=max_iter,
        maxDepth=max_depth,
        seed=42
    )
    stages.append(gbt)
    
    return Pipeline(stages=stages)
