import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils

// load csv filter header
val rawDatain = sc.textFile("/Users/wy/Desktop/advance_Spark/train.csv")
val header = rawDatain.first
val rawData = rawDatain.filter(_ != header)

import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._

val data = rawData.map{ line =>
	val value = line.split(",")
	val values = Array(value(1).toDouble,value(2).toDouble,value(3).toDouble,value(4).toDouble,value(5).toDouble,value(6).toDouble,value(7).toDouble,value(8).toDouble,value(9).toDouble,value(10).toDouble,value(11).toDouble)
	val featureVector = Vectors.dense(values.init)
	val label = values.last-1
	LabeledPoint(label,featureVector)
}

val Array(trainData, testData) = data.randomSplit(Array(0.7, 0.3))

trainData.cache()
testData.cache()

val categoricalFeaturesInfo = Map[Int, Int]()
val numTrees = 3 // Use more in practice.
val featureSubsetStrategy = "auto" // Let the algorithm choose.
val impurity = "variance"
val maxDepth = 4
val maxBins = 32

val model = RandomForest.trainRegressor(trainData, categoricalFeaturesInfo,
  numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

val labelsAndPredictions = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}

val testMSE = labelsAndPredictions.map{ case(v, p) => math.pow((v - p), 2)}.mean()
println("Test Mean Squared Error = " + testMSE)
println("Learned regression forest model:\n" + model.toDebugString)

// Save and load model
model.save(sc, "myModelPath")
val sameModel = RandomForestModel.load(sc, "myModelPath")


