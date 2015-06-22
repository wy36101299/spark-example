// kaggle bike sharing https://www.kaggle.com/c/bike-sharing-demand

// Decision tree Regression
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._

// load csv filter header
val rawDatain = sc.textFile("/Users/wy/Desktop/advance_Spark/train.csv")
val header = rawDatain.first
val rawData = rawDatain.filter(_ != header)

// Vectors.dense : https://spark.apache.org/docs/latest/mllib-data-types.html
val data = rawData.map{ line =>
	val value = line.split(",")
	val values = Array(value(1).toDouble,value(2).toDouble,value(3).toDouble,value(4).toDouble,value(5).toDouble,value(6).toDouble,value(7).toDouble,value(8).toDouble,value(9).toDouble,value(10).toDouble,value(11).toDouble)
	val featureVector = Vectors.dense(values.init)
	val label = values.last-1
	LabeledPoint(label,featureVector)
}

val Array(trainData, testData) = data.randomSplit(Array(0.8, 0.2))

trainData.cache()
testData.cache()

import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.model._
import org.apache.spark.rdd._

// https://spark.apache.org/docs/latest/mllib-decision-tree.html
val categoricalFeaturesInfo = Map[Int, Int]()
val impurity = "variance"
val maxDepth = 5
val maxBins = 32

val model = DecisionTree.trainRegressor(trainData, categoricalFeaturesInfo, impurity,
  maxDepth, maxBins)

val labelsAndPredictions = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
  // (point.features,point.label, prediction)
}

val testMSE = labelsAndPredictions.map{ case(v, p) => math.pow((v - p), 2)}.mean()
println("Test Mean Squared Error = " + testMSE)
println("Learned regression tree model:\n" + model.toDebugString)

// turning 
val evaluations =
  for (impurity <- Array("variance");
       maxDepth    <- Array(3, 5);
       maxBins     <- Array(16, 32))
    yield {
      val model = DecisionTree.trainRegressor(
        trainData,Map[Int,Int](), impurity, maxDepth, maxBins)

	  val labelsAndPredictions = testData.map { point =>
	    val prediction = model.predict(point.features)
	    (point.label, prediction)
	    // (point.features,point.label, prediction)
	  }
      val testMSE = labelsAndPredictions.map{ case(v, p) => math.pow((v - p), 2)}.mean()

      ((maxDepth, maxBins), testMSE)

    }
evaluations.sortBy(_._2).foreach(println)

// Save model
model.save(sc, "/Users/wy/Desktop/advance_Spark/model")
// Load model
val sameModel = DecisionTreeModel.load(sc, "/Users/wy/Desktop/advance_Spark/model")

