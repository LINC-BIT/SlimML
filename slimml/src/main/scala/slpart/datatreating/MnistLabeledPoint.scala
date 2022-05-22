package slpart.datatreating

import java.nio.ByteBuffer
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import scala.collection.mutable.ArrayBuffer

object MnistLabeledPoint {
  /**
    * 加载image - 即是features
    * @param filePath
    * @return
    */
  def loadImageData(filePath: String): Array[Array[Int]] ={
    val bytes = BytesDataLoad.load(filePath)

    val magicNumber = ByteBuffer.wrap(bytes.take(4)).getInt()
    val numberOfImages = ByteBuffer.wrap(bytes.slice(4,8)).getInt()
    val numberOfRows = ByteBuffer.wrap(bytes.slice(8,12)).getInt()
    val numberOfCols = ByteBuffer.wrap(bytes.slice(12,16)).getInt()
    val sizeOfImage = numberOfRows*numberOfCols

    println("Magic Number: %d".format(magicNumber))
    println("Number of images: %d".format(numberOfImages))
    println("Size  of image: %d x %d".format(numberOfRows,numberOfCols))

    val imageDataBuffer = new ArrayBuffer[Array[Int]]
    var i: Int =0
    while( i < numberOfImages){
      imageDataBuffer += bytes.slice(16 + i * sizeOfImage, 16 + (i+1) * sizeOfImage).map( _ & 0xff)
      i += 1
    }
    imageDataBuffer.toArray
  }

  /**
    * 加载label文件
    * @param filePath
    * @return
    */
  def loadLabelData(filePath: String): Array[Int] = {
    val bytes = BytesDataLoad.load(filePath)

    val magicNumber = ByteBuffer.wrap(bytes.take(4)).getInt()
    val numberOfImages = ByteBuffer.wrap(bytes.slice(4,8)).getInt()

    println("Magic Number: %d".format(magicNumber))
    println("Number of Images: %d".format(numberOfImages))

    val labelDataBuffer = bytes.drop(8).map( _ & 0xff).map(_ + 1)
    labelDataBuffer
  }

  /**
    * 将feature和对应的label组合成LabeledPoint
    * @param imageData
    * @param labelData
    * @return
    */
  def combineData(imageData:Array[Array[Int]],labelData:Array[Int]): Array[LabeledPoint] ={
    val dataSet = labelData.zip(imageData).map{ x =>
      new LabeledPoint(x._1.toDouble,Vectors.dense(x._2.map(_.toDouble)))
    }
    dataSet
  }

  def loadTrain(featureFile: String, labelFile: String) = {
    val imageArray = loadImageData(featureFile)
    val labelArray = loadLabelData(labelFile)
    combineData(imageArray,labelArray)
  }
  def loadTest(featureFile: String,labelFile: String) = {
    val imageArray = loadImageData(featureFile)
    val labelArray = loadLabelData(labelFile)
    combineData(imageArray,labelArray)
  }

}
