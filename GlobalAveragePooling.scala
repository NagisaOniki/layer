package layer
import breeze.linalg._
import pll.utils.RichArray._
class GlobalAveragePooling(val OH:Int , val OW:Int , val OC:Int) extends Layer {
  override type T = Float
  def forward(Z:Array[T])={
    var ave = new Array[T](OC)
    for(i<-0 until OC){
      for(m<-0 until OH ; n<-0 until OW){
        ave(i) += Z(i*OH*OW + m*OW+n) / (OW*OH)
      }
    }
    ave
  }
  override def forward(Z:Array[Array[T]])={
    Z.map(forward)
  }
  def backward(A:Array[T])={
    A
  }
  override def backward(A:Array[Array[T]])={
    A.reverse.map(backward).reverse
  }
  def update(){
  }
  def reset(){
  }
}
