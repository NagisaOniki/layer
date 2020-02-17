package layer
import breeze.linalg._
import pll.utils.RichArray._
class Pooling(val BW:Int, val IC:Int, val IH:Int, val IW:Int) extends Layer {
  override type T = Float
  val OH = IH / BW
  val OW = IW / BW
  val OC = IC
  var masks = List[Array[T]]()
  def push(x:Array[T]) = { masks ::= x; x }
  def pop() = { val mask = masks.head; masks = masks.tail; mask }
  def iindex(i:Int, j:Int, k:Int) = i * IH * IW + j * IW + k
  def oindex(i:Int, j:Int, k:Int) = i * OH * OW + j * OW + k
  def forward(X:Array[T]) = {
    val mask = push(Array.ofDim[T](IC * IH * IW))
    val Z = Array.ofDim[T](OC * OH * OW)
    for(i <- 0 until OC; j <- 0 until OH; k <- 0 until OW) {
      var v = Float.NegativeInfinity
      var row_max = -1
      var col_max = -1
      for(m <- 0 until BW; n <- 0 until BW if v < X(iindex(i,j*BW+m,k*BW+n))) {
        row_max = j*BW+m
        col_max = k*BW+n
        v = X(iindex(i,j*BW+m,k*BW+n))
      }
      mask(iindex(i,row_max,col_max)) = 1
      Z(oindex(i,j,k)) = v
    }
    Z
  }
  override def forward(x:Array[Array[T]])={
    x.map(forward)
  }
  def backward(d:Array[T]) = {
    val mask = pop()
    val D = Array.ofDim[T](mask.size)
    for(i <- 0 until OC; j <- 0 until OH; k <- 0 until OW) {
      for(m <- 0 until BW; n <- 0 until BW if mask(iindex(i,j*BW+m,k*BW+n)) == 1) {
        D(iindex(i,j*BW+m,k*BW+n)) = d(oindex(i,j,k))
      }
    }
    D
  }
  override def backward(d:Array[Array[T]])={
    d.reverse.map(backward).reverse
  }
  def update() {
    reset()
  }
  def reset() {
    masks = List[Array[T]]()
  }
}
