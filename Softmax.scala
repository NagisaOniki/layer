package layer
import breeze.linalg._
import pll.utils.RichArray._
class Softmax() extends Layer {
  override type T = Float
  var ys = List[Array[T]]()
  var ys1 = List[DenseVector[T]]()
  var xsum = 0f
  def push(y:Array[T]) = { ys ::= y; y }
  def pop() = { val y = ys.head; ys = ys.tail; y }
  def pushd(x:DenseVector[T]) = { ys1 ::= x; x }
  def popd() = { val x = ys1.head; ys1 = ys1.tail; x }
  def init() = { }
  override def save(pass : String) = {
  }
  def loadparameter(pass:String) = {
  }
  def prof() = {
    "LeakyReLU()"
  }
  def load_h(h:DenseVector[T]) = {
  }
  override def forward(x:Array[Array[T]]) = {
    var xs = Array.ofDim[T](x.size , x(0).size)
    for(i <- 0 until x.size){
      xs(i) = forward(x(i))
    }
    xs
  }
  override def backward(d:Array[Array[T]]) = {
    val ds = d.reverse.map(backward).reverse
    ds
  }
  def forward(x:Array[T]) = {
    val xs = x.map(a => a - x.max)
    val xsDouble = xs.map(_.toDouble)
    val xexp = (xsDouble.map(math.exp)).map(_.toFloat)
    push(xexp)
    xsum = xexp.sum
    val dsum = 1f / xsum
    var y = new Array[T](x.size)
    for(i <- 0 until x.size){
      y(i) = xexp(i) * dsum
    }
    y
  }
  def backward(d:Array[T]) = {
    val x = pop()
    var y = new Array[T](d.size)
    val dx = (0 until d.size).map(i => d(i) * x(i)).sum
    for(i <- 0 until d.size){
      val a = ((d(i) / xsum) - (1f / xsum * xsum) * dx)
      y(i) = a * x(i)
    }
    y
  }
  def forward(x:DenseVector[T]) = {
    val xs = x.map(a => a - max(x))
    val xsDouble = xs.map(_.toDouble)
    val xexp = (xsDouble.map(math.exp)).map(_.toFloat)
    pushd(xexp)
    xsum = sum(xexp)
    val dsum = 1f / xsum
    var y = DenseVector.zeros[T](x.size)
    for(i <- 0 until x.size){
      y(i) = xexp(i) * dsum
    }
    y
  }
  def backward(d:DenseVector[T]) = {
    val x = popd()
    var y = DenseVector.zeros[T](d.size)
    val dx = sum((0 until d.size).map(i => d(i) * x(i)))
    for(i <- 0 until d.size){
      val a = ((d(i) / xsum) - (1f / xsum * xsum) * dx)
      y(i) = a * x(i)
    }
    y
  }
  def update() {
    reset()
  }
  def update_RMSProp() = {
  }
  def update_Adam() = {
  }
  def reset() {
    ys = List[Array[T]]()
    ys1 = List[DenseVector[T]]()
  }
}
