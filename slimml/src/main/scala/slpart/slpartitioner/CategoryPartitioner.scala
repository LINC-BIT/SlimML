package slpart.slpartitioner

import  org.apache.spark.Partitioner

class CategoryPartitioner(categoryNum: Int) extends Partitioner {
  override def numPartitions: Int = categoryNum

  override def getPartition(key: Any): Int = {
    val k = key.asInstanceOf[Int]
    k % categoryNum
  }
}
