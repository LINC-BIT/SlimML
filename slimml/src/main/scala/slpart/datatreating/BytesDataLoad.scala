package slpart.datatreating

import java.nio.ByteBuffer
import java.nio.file.{Paths,Files}
import com.intel.analytics.bigdl.utils.File

/**
  * 提供一个可以加载
  * 本地二进制文件和hdfs二进制文件
  * 的接口
  */
object BytesDataLoad {
  private val hdfsPrefix: String = "hdfs:"

  /**
    * 加载二进制文件
    * 返回二进制数组
    * @param filePath
    * @return
    */
  def load(filePath: String): Array[Byte] = {
    val buffer = if(filePath.startsWith(hdfsPrefix)){
      ByteBuffer.wrap(File.readHdfsByte(filePath))
    }
    else{
      ByteBuffer.wrap(Files.readAllBytes(Paths.get(filePath)))
    }
    buffer.array()
  }
}
