package slpart.datatreating

import org.apache.spark.mllib.regression.LabeledPoint
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.collection.mutable.ArrayBuffer

class CustomArySampleIter(iter: Iterator[(Array[ArrayBuffer[LabeledPoint]],Long)],maxOrigNum: Int,
                          needCid: Boolean = false,scale: Double = 1.0,ignorePadding: Option[Double] = None)
extends Iterator[Sample[Float]]{
  override def hasNext: Boolean = {
    iter.hasNext
  }

  override def next(): Sample[Float] = {
    val p = iter.next()

    val feaAry = new ArrayBuffer[Double]()
    val labAry = new ArrayBuffer[Double]()

    feaAry ++= p._1.head.head.features.toArray
    val nl = if(!needCid) p._1.head.head.label else p._1.head.head.label + (p._2 + 2) * scale
    labAry += (nl + 0.25).toInt

    var curOrig = p._1.last.length
    for(elem <- p._1.last){
      feaAry ++= elem.features.toArray
      labAry += elem.label
    }
    while(curOrig < maxOrigNum){
      val idx = (new scala.util.Random()).nextInt(p._1.last.length) % p._1.last.length
      feaAry ++= p._1.last(idx).features.toArray
      if(ignorePadding.isEmpty)labAry += p._1.last(idx).label
      else labAry += ignorePadding.get
      curOrig += 1
    }
    val f = feaAry.toArray
    val l = labAry.toArray
    Sample(Tensor[Float](T(f.head,f.tail: _*)),Tensor[Float](T(l.head,l.tail: _*)))
  }
}
