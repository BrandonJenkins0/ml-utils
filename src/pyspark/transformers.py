import pyspark.sql.functions as F
from pyspark.ml import Transformer
from pyspark.sql.window import Window


class SupervisedTSData(Transformer):
    def __init__(
        self,
        *,
        targetCol,
        timeCol,
        groupCol,
        forecastHorizon=1,
        dropNa=True,
        makeRowLag1=True,
    ):
        super(SupervisedTSData, self).__init__()
        self.targetCol = targetCol
        self.timeCol = timeCol
        self.groupCol = groupCol if isinstance(groupCol, list) else [groupCol]
        self.dropNa = dropNa
        self.makeRowLag1 = makeRowLag1
        self.forecastHorizon = (
            forecastHorizon if self.makeRowLag1 else forecastHorizon - 1
        )

    def _transform(self, dataset):
        w = Window.orderBy(self.timeCol)

        if self.groupCol[0]:
            w = w.partitionBy(*self.groupCol)

        dataset = dataset.withColumn(
            self.targetCol, F.lead(self.targetCol, self.forecastHorizon).over(w)
        )

        if self.dropNa:
            dataset = dataset.dropna()

        return dataset


class DateFeatures(Transformer):
    def __init__(self, timeCol):
        self.timeCol = timeCol

    def _transform(self, dataset):
        dataset = dataset.withColumn(
            "month", F.month(self.timeCol).alias(self.timeCol)
        )
        dataset = dataset.withColumn(
            "year", F.year(self.timeCol).alias(self.timeCol)
        )
        dataset = dataset.withColumn(
            "dayofweek", F.dayofweek(self.timeCol).alias(self.timeCol)
        )
        dataset = dataset.withColumn(
            "dayofmonth", F.dayofmonth(self.timeCol).alias(self.timeCol)
        )
        return dataset


class LagFeatures(Transformer):
    def __init__(
        self,
        *,
        lagDict,
        timeCol,
        targetCol,
        groupCol=None,
        dropOgCols=False,
        dropNa=False,
        isRowLag1=True,
    ):
        super(SupervisedTSData, self).__init__()
        self.lagDict = lagDict
        self.timeCol = timeCol
        self.targetCol = targetCol
        self.groupCol = groupCol if isinstance(groupCol, list) else [groupCol]
        self.dropOgCols = dropOgCols
        self.dropNa = dropNa
        self.isRowLag1 = isRowLag1

    def _transform(self, dataset):
        w = Window.orderBy(self.timeCol)

        if self.groupCol[0]:
            w = w.partitionBy(*self.groupCol)

        for col, lags in self.lagDict.items():
            if self.isRowLag1:
                for lag in lags:
                    dataset = dataset.withColumn(
                        f"lag_{col}_{lag}", F.lag(col, lag - 1).over(w)
                    )
            else:
                for lag in lags:
                    dataset = dataset.withColumn(
                        f"lag_{col}_{lag}", F.lag(col, lag).over(w)
                    )

        if self.dropOgCols:
            dataset = dataset.select(
                [
                    col
                    for col in dataset.columns
                    if col not in self.lagDict.keys() or col == self.targetCol
                ]
            )

        if self.dropNa:
            dataset = dataset.dropna()

        return dataset
