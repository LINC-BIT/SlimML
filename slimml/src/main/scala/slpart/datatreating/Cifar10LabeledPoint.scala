package slpart.datatreating

import java.nio.ByteBuffer
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import scala.collection.mutable.ArrayBuffer

object Cifar10LabeledPoint {
  private val itemEachFile: Int = 10000
  private val labelSize: Int = 1
  private val featureSize: Int = 3072 //3*32*32
  private val onePointSize: Int = labelSize + featureSize

  /**
    * 以统一的格式加载feature and label
    * @param filePath
    * @return
    */
  def load(filePath:String):ArrayBuffer[LabeledPoint] = {
    val bytes = BytesDataLoad.load(filePath)

    val labelArray = new ArrayBuffer[Int]
    val featureArray = new ArrayBuffer[Array[Int]]

    val newData = bytes.map(_ & 0xff)
    for(cur <- 0 until itemEachFile){
      labelArray += newData(cur*onePointSize) + 1
      featureArray += newData.slice(cur*onePointSize + labelSize,cur*onePointSize + labelSize + featureSize)
    }

    val pointArray = new ArrayBuffer[LabeledPoint]
    for(cur <- 0 until(itemEachFile)){
      pointArray += LabeledPoint(labelArray(cur).toDouble,Vectors.dense(featureArray(cur).map(_.toDouble)))
    }
    pointArray
  }

  def loadTrain(dataFile: String) = {
    val allFiles = Array(
      dataFile + "/data_batch_1.bin",
      dataFile + "/data_batch_2.bin",
      dataFile + "/data_batch_3.bin",
      dataFile + "/data_batch_4.bin",
      dataFile + "/data_batch_5.bin"
    )
    allFiles.flatMap(load(_))
  }

  def loadTest(dataFile: String) = {
    val testFile = dataFile + "/test_batch.bin"
    load(testFile).toArray
  }
}
