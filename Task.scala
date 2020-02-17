package layer
import breeze.linalg._
import pll.utils.RichArray._
import scala.util.control.Breaks
object task{
  type T = Float

  def NaNfound(path:String){
    val source = scala.io.Source.fromFile(path).getLines.toList
    val ln = source(0).split(",").map(_.toInt)
    val loss = source(1).split(",")
    val b = new Breaks
    for(i<-0 until ln.size){
      if(loss(i) == "NaN"){
        println("lnNumber"+(i+1)+"~:NaN")
        b.break
      }
    }
  }

  //save
  def save(fn:String,input:Array[T]){
    val save = new java.io.PrintWriter(fn)
    for(i<-0 until input.size){
      save.print(i+1)
      if(i != input.size-1){save.print(",")}
    }
    save.println("")
//      (1 to input.size).map(i => i).toArray.mkString(",")+"\n" //学習回数のプリント
    for(i<-0 until input.size){
      save.print(input(i))
      if(i != input.size-1){save.print(",")}                             
    }
    save.close
  }

  def saveDCGAN(
    fn:String,
    ln:Int,
    lossG:Array[T],
    lossD:Array[T],
    correctDx:Array[T],
    correctDz:Array[T],
    time:Float
  ){
    val save = new java.io.PrintWriter(fn)
    for(i<-0 until ln){
      save.println(i+":lossG:"+lossG(i)+"lossD:"+lossD(i)+"/CorrectDx:"+correctDx(i)+"CorrectDz:"+correctDz(i))
    }
    save.println("Time"+time)
  }

  def rgbDist(path0:String,path1:String)={
    val read0 = Image.read(path0)
    val read1 = Image.read(path1)
    val image0 = read0.map(_.map(_.map(_.toFloat)))
    val image1 = read1.map(_.map(_.map(_.toFloat)))
    val H = image0.size
    val W = image0(0).size
    val C = image0(0)(0).size
    var dist = 0f
    for(i<-0 until H ; j<-0 until W ; k<-0 until C){
      dist += (image0(i)(j)(k) - image1(i)(j)(k)) * (image0(i)(j)(k) - image1(i)(j)(k))
    }
    dist
  }

  def rgbAve(path0:String)={
    val read = Image.read(path0)
    val H = read.size
    val W = read(0).size
    val size = (H*W).toDouble
    var R = 0d
    var G = 0d
    var B = 0d
    for(i<-0 until H ; j<-0 until W){
      R += read(i)(j)(0) / size
      G += read(i)(j)(1) / size
      B += read(i)(j)(2) / size
    }
    (R,G,B)
    //(R/H*W , G/H*W , B/H*W)
  }

  def rgbAveBorder(path0:String)={
    val Z = Image.read(path0)
//    val V = Image.read("border4x4.png")
    val V = Image.read("GraduationResearch/border32x32.png")
    val H = Z.size
    val W = Z(0).size
    var whiteSize = 0f
    var blackSize = 0f
    var R_white = 0f
    var G_white = 0f
    var B_white = 0f
    var R_black = 0f
    var G_black = 0f
    var B_black = 0f
    for(i<-0 until H ; j<-0 until W){
      if(V(i)(j)(0) > 250 && V(i)(j)(1) > 250 && V(i)(j)(2) > 250){ //white
        R_white += Z(i)(j)(0)
        G_white += Z(i)(j)(1)
        B_white += Z(i)(j)(2)
        whiteSize += 1f
      }else if(V(i)(j)(0) < 5 && V(i)(j)(1) < 5 && V(i)(j)(2) < 5){
        R_black += Z(i)(j)(0)
        G_black += Z(i)(j)(1)
        B_black += Z(i)(j)(2)
        blackSize += 1f
      }
    }
    ((R_white/whiteSize,G_white/whiteSize,B_white/whiteSize),(R_black/blackSize,G_black/blackSize,B_black/blackSize))
  }

  def printRgb(path:String){
    val read = Image.read(path)
    val H = read.size
    val W = read(0).size
    for(i<-0 until H ; j<-0 until W){
      print("("+read(i)(j)(0)+","+read(i)(j)(1)+","+read(i)(j)(2)+") , ")
    }
  }


}
