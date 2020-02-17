package layer 
import breeze.linalg._
import pll.utils.RichArray._
class ReLU() extends Layer {
  override type T = Float
  var ys = List[Array[T]]()
  def push(y:Array[T]) = { ys ::= y; y }
  def pop() = { val y = ys.head; ys = ys.tail; y }
  def forward(x:Array[T]) = {
    push(x.map(a => math.max(a,0)))
  }
  override def forward(x:Array[Array[T]]) = {
    x.map(forward)
  }
  def backward(d:Array[T]) = {
    val y = pop()
      (0 until d.size).map(i => if(y(i) > 0) d(i) else 0f).toArray
  }
  override def backward(d:Array[Array[T]]) = {
    d.reverse.map(backward).reverse
  }
  def update() {
    //reset()
  }
  def reset() {
    ys = List[Array[T]]()
  }
}
