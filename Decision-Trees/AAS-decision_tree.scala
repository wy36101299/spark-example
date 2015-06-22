// Chapter 4. Predicting Forest Cover with Decision Trees
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._

val rawData = sc.textFile("/Users/wy/Desktop/advance_Spark/covtype.data")

// Vectors.dense : https://spark.apache.org/docs/latest/mllib-data-types.html
val data = rawData.map{ line =>
	val values = line.split(",").map(_.toDouble)
	val featureVector = Vectors.dense(values.init)
	val label = values.last-1
	LabeledPoint(label,featureVector)
}

val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))

trainData.cache()
cvData.cache()
testData.cache()

import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.model._
import org.apache.spark.rdd._

// get MulticlassMetrics
def getMetrics(model: DecisionTreeModel, data: RDD[LabeledPoint]):
    MulticlassMetrics = {
  val predictionsAndLabels = data.map(example =>
    (model.predict(example.features), example.label)
  )
  new MulticlassMetrics(predictionsAndLabels)
}


val model = DecisionTree.trainClassifier(trainData, 7, Map[Int,Int](), "gini", 4, 100)

val metrics = getMetrics(model, cvData)

metrics.confusionMatrix
// 14019.0  6630.0   15.0    0.0    0.0  1.0   391.0
// 5413.0   22399.0  438.0   16.0   0.0  3.0   50.0
// 0.0      457.0    2999.0  73.0   0.0  12.0  0.0
// 0.0      1.0      163.0   117.0  0.0  0.0   0.0
// 0.0      872.0    40.0    0.0    0.0  0.0   0.0
// 0.0      500.0    1138.0  36.0   0.0  48.0  0.0
// 1091.0   41.0     0.0     0.0    0.0  0.0   891.0

metrics.precision
// 0.7030630195577938

(0 until 7).map(
  cat => (metrics.precision(cat), metrics.recall(cat))
).foreach(println)

// (0.6805931840866961,0.6809492105763744)
// (0.7297560975609756,0.7892237892589596)
// (0.6376224968044312,0.8473952434881087)
// (0.5384615384615384,0.3917910447761194)
// (0.0,0.0)
// (0.7083333333333334,0.0293778801843318)
// (0.6956168831168831,0.42828585707146427)

import org.apache.spark.rdd._

def classProbabilities(data: RDD[LabeledPoint]): Array[Double] = {
  val countsByCategory = data.map(_.label).countByValue()
  val counts = countsByCategory.toArray.sortBy(_._1).map(_._2)
  counts.map(_.toDouble / counts.sum)
}

val trainPriorProbabilities = classProbabilities(trainData)
val cvPriorProbabilities = classProbabilities(cvData)
trainPriorProbabilities.zip(cvPriorProbabilities).map {
  case (trainProb, cvProb) => trainProb * cvProb
}.sum
// 0.37737764750734776 Random guessing achieves 37% accuracy

val evaluations =
  for (impurity <- Array("gini", "entropy");
       depth    <- Array(1, 20);
       bins     <- Array(10, 300))
    yield {
      val model = DecisionTree.trainClassifier(
        trainData, 7, Map[Int,Int](), impurity, depth, bins)
      val predictionsAndLabels = cvData.map(example =>
        (model.predict(example.features), example.label)
      )
      val accuracy =
        new MulticlassMetrics(predictionsAndLabels).precision
      ((impurity, depth, bins), accuracy)
    }

evaluations.sortBy(_._2).reverse.foreach(println)

// ((entropy,20,300),0.9125545571245186)
// ((gini,20,300),0.9042533162173727)
// ((gini,20,10),0.8854428754813863)
// ((entropy,20,10),0.8848951647411211)
// ((gini,1,300),0.6358065896448438)
// ((gini,1,10),0.6355669661959777)
// ((entropy,1,300),0.4861446298673513)
// ((entropy,1,10),0.4861446298673513)

val model = DecisionTree.trainClassifier(
  trainData.union(cvData), 7, Map[Int,Int](), "entropy", 20, 300)

val data = rawData.map { line =>
  val values = line.split(',').map(_.toDouble)
  val wilderness = values.slice(10, 14).indexOf(1.0).toDouble
  val soil = values.slice(14, 54).indexOf(1.0).toDouble
  val featureVector =
    Vectors.dense(values.slice(0, 10) :+ wilderness :+ soil)
  val label = values.last - 1
  LabeledPoint(label, featureVector)
}

val evaluations =
  for (impurity <- Array("gini", "entropy");
       depth    <- Array(10, 20, 30);
       bins     <- Array(40, 300))
    yield {
      val model = DecisionTree.trainClassifier(
        trainData, 7, Map(10 -> 4, 11 -> 40),
        impurity, depth, bins)
      val trainAccuracy = getMetrics(model, trainData).precision
      val cvAccuracy = getMetrics(model, cvData).precision
      ((impurity, depth, bins), (trainAccuracy, cvAccuracy))
    }

// ((entropy,30,300),(0.9996922984231909,0.9438383977425239))
// ((entropy,30,40),(0.9994469978654548,0.938934581368939))
// ((gini,30,300),(0.9998622874061833,0.937127912178671))
// ((gini,30,40),(0.9995180059216415,0.9329467634811934))
// ((entropy,20,40),(0.9725865867933623,0.9280773598540899))
// ((gini,20,300),(0.9702347139020864,0.9249630062975326))
// ((entropy,20,300),(0.9643948392205467,0.9231391307340239))
// ((gini,20,40),(0.9679344832334917,0.9223820503114354))
// ((gini,10,300),(0.7953203539213661,0.7946763481193434))
// ((gini,10,40),(0.7880624698753701,0.7860215423792973))
// ((entropy,10,40),(0.78206336500723,0.7814790598437661))
// ((entropy,10,300),(0.7821903188046547,0.7802746137169208))

// Random Decision Forests
val forest = RandomForest.trainClassifier(
  trainData, 7, Map(10 -> 4, 11 -> 40), 20,
  "auto", "entropy", 30, 300)

val input = "2709,125,28,67,23,3224,253,207,61,6094,0,29"
val vector = Vectors.dense(input.split(',').map(_.toDouble))
forest.predict(vector)

