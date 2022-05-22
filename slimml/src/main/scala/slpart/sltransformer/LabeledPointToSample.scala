package slpart.sltransformer
import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.apache.spark.mllib.regression.LabeledPoint

object LabeledPointToSample{
  def apply(): LabeledPointToSample = {
    new LabeledPointToSample()
  }
}

class LabeledPointToSample extends Transformer[LabeledPoint,Sample[Float]]{
  private val featureBuffer = Tensor[Float]()
  private val labelBuffer = Tensor[Float](1)
  private val featureSize = new Array[Int](1)
  override def apply(prev: Iterator[LabeledPoint]): Iterator[Sample[Float]] = {
    prev.map(lp =>{
      labelBuffer.storage().array()(0) = lp.label.asInstanceOf[Float]
      featureSize(0) = lp.features.toArray.length
      featureBuffer.set(Storage(lp.features.toArray.map(_.asInstanceOf[Float])),sizes = featureSize)
      Sample(featureBuffer,labelBuffer)
    })
  }
}