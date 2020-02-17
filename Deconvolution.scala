package layer
import breeze.linalg._
import pll.utils.RichArray._
class Deconvolution(val IH:Int,val IW:Int,val IC:Int,val S:Int)extends Layer{
  override type T = Float
  val OH = S * IH + S - 1
  val OW = S * IW + S - 1
  val OC = IC
  def iindex(i:Int , j:Int , k:Int) = i * IH * IW + j * IW + k
  def oindex(i:Int , j:Int , k:Int) = i * OH * OW + j * OW + k
  def forward(X:Array[T]) = {
    val Z = Array.ofDim[T](OC * OH * OW)
    for(i<-0 until IC ; j<-0 until IH ; k<-0 until IW){
      Z(i*OH*OW + (S-1+j*S)*OW + S-1+k*S) = X(i*IH*IW + j*IW + k)
    }
    Z
  }
  override def forward(x:Array[Array[T]])={
    x.map(forward)
  }
  def backward(d:Array[T]) = {
    val D = Array.ofDim[T](IC * IH * IW)
    for(i<-0 until IC ; j<-0 until IH ; k<-0 until IW){
      D(i*IH*IW + j*IW + k) = d(i*OH*OW + (S-1+j*S)*OW + S-1+k*S)
    }
    D
  }
  override def backward(d:Array[Array[T]])={
    d.reverse.map(backward).reverse
  }
  def update(){}
  def reset(){}
}
