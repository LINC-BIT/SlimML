package slpart.models.lenet

object TestBooleanOptions {
  import  Utils._

  def main(args: Array[String]): Unit = {
    trainParser.parse(args,new TrainParams()).map(param => {
      System.out.println(s"useComp: ${param.useComp}")
      System.out.println(s"onlyComp: ${param.onlyComp}")
      System.out.println(s"getGradient: ${param.getGradient}")
      System.out.println(s"graphModel: ${param.graphModel}")
    })
  }
}
