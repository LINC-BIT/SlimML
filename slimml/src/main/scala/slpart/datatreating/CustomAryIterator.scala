package slpart.datatreating

import org.apache.spark.mllib.regression.LabeledPoint
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.collection.mutable.ArrayBuffer

class CustomAryIterator
(iter: Iterator[(LabeledPoint,Array[LabeledPoint])],maxOrigNum: Int,ignorePadding: Option[Double] = None)
  extends Iterator[Sample[Float]] {
  override def hasNext: Boolean = {
    iter.hasNext
  }

  override def next(): Sample[Float] = {
    val p = iter.next()

    val feaAry = new ArrayBuffer[Double]()
    val labAry = new ArrayBuffer[Double]()

    feaAry ++= p._1.features.toArray
    labAry += (p._1.label + 0.25).toInt

    var curOrig = p._2.length
    for(elem <- p._2){
      feaAry ++= elem.features.toArray
      labAry += elem.label
    }
    while(curOrig < maxOrigNum){
      val idx = (new scala.util.Random()).nextInt(p._2.length) % p._2.length
      feaAry ++= p._2(idx).features.toArray
      if(ignorePadding.isEmpty) labAry += p._2(idx).label
      else labAry += ignorePadding.get
      curOrig += 1
    }
    val f = feaAry.toArray
    val l = labAry.toArray
    Sample(Tensor[Float](T(f.head,f.tail: _*)),Tensor[Float](T(l.head,l.tail: _*)))
  }
}

