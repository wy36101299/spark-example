import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
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

// Train a GradientBoostedTrees model.
//  The defaultParams for Regression use SquaredError by default.
val boostingStrategy = BoostingStrategy.defaultParams("Regression")
boostingStrategy.numIterations = 3 // Note: Use more iterations in practice.
boostingStrategy.treeStrategy.maxDepth = 5
//  Empty categoricalFeaturesInfo indicates all features are continuous.
boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

val model = GradientBoostedTrees.train(trainData, boostingStrategy)

// Evaluate model on test instances and compute test error
val labelsAndPredictions = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}

val testMSE = labelsAndPredictions.map{ case(v, p) => math.pow((v - p), 2)}.mean()
println("Test Mean Squared Error = " + testMSE)
println("Learned regression GBT model:\n" + model.toDebugString)

// Save and load model
model.save(sc, "myModelPath")
val sameModel = GradientBoostedTreesModel.load(sc, "myModelPath")
