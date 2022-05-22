package slpart.sltransformer

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.numeric.NumericFloat
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector,Vectors}

object SampleToLabeledPoint{
  def apply(): SampleToLabeledPoint = {
    new SampleToLabeledPoint()
  }
}

class SampleToLabeledPoint extends Transformer[Sample[Float],LabeledPoint]{
  override def apply(prev: Iterator[Sample[Float]]): Iterator[LabeledPoint] = {
    prev.map(sample => {
      val shape = sample.feature().size()
      var prd = 1
      shape.foreach(x => prd = prd * x)
      val features = sample.feature().reshape(Array(1,prd)).squeeze().toArray().map(_.asInstanceOf[Double])
      val label = sample.label().squeeze().toArray()(0).asInstanceOf[Double]
      LabeledPoint(label,Vectors.dense(features))
    })
  }
}
