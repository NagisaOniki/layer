package layer
import layer.Adam._
import breeze.linalg._
import pll.utils.RichArray._
class AffineT(val xn:Int, val yn:Int, val eps:T = 0.002f, val rho1:T = 0.5f, val rho2:T = 0.999f) extends Layer {
  override type T = Float
  var W = Array.ofDim[T](xn * yn)
  var b = Array.ofDim[T](yn)
  var dW = Array.ofDim[T](xn * yn)
  var db = Array.ofDim[T](yn)
  var n = 0f
  def windex(i:Int, j:Int) = i * xn + j
  var xs = List[Array[T]]()
  def push(x:Array[T]) = { xs ::= x; x }
  def pop() = { val x = xs.head; xs = xs.tail; x }
  def forward(x:Array[T]) = {
    push(x)
    val y = Array.ofDim[T](yn)
    for(i <- 0 until yn) {
      for(j <- 0 until xn) {
        y(i) += W(windex(i,j)) * x(j)
      }
      y(i) += b(i)
    }
    y
  }
  override def forward(x:Array[Array[T]])={
    x.map(forward)
  }
  def backward(d:Array[T]) = {
    val x = pop()
    n += 1f
    for(i <- 0 until yn; j <- 0 until xn) {
      dW(windex(i,j)) += d(i) * x(j)
    }
    for(i <- 0 until yn) {
      db(i) += d(i)
    }
    val dx = Array.ofDim[T](xn)
    for(j <- 0 until yn; i <- 0 until xn) {
      dx(i) += W(windex(j,i)) * d(j)
    }
    dx
  }
  override def backward(d:Array[Array[T]])={
    d.reverse.map(backward).reverse
  }
  def update() {
    for(i <- 0 until dW.size) {
      dW(i) /= n
    }
    for(i <- 0 until db.size) {
      db(i) /= n
    }
    update_adam()
    reset()
  }
  var adam_W = new Adam_BNa(W.size, eps, rho1, rho2)
  var adam_b = new Adam_BNa(b.size, eps, rho1, rho2)
  def update_adam() {
    adam_W.update(W,dW)
    adam_b.update(b,db)
  }
  var lr = 0.001f
  def update_sgd() {
    for(i <- 0 until W.size) {
      W(i) -= lr * dW(i)
    }
    for(i <- 0 until b.size) {
      b(i) -= lr * db(i)
    }
  }
  def reset() {
    for(i <- 0 until dW.size) {
      dW(i) = 0f
    }
    for(i <- 0 until db.size) {
      db(i) = 0f
    }
    xs = List[Array[T]]()
    n = 0
  }
  override def save(fn:String) {
    val pw = new java.io.PrintWriter(fn)
    for(i <- 0 until W.size) {
      pw.write(W(i).toString)
      if(i != W.size - 1) {
        pw.write(",")
      }
    }
    pw.write("\n")
    for(i <- 0 until b.size) {
      pw.write(b(i).toString)
      if(i != b.size - 1) {
        pw.write(",")
      }
    }
    pw.write("\n")
    pw.close()
  }
  override def load(fn:String) {
    val f = scala.io.Source.fromFile(fn).getLines.toArray
    W = f(0).split(",").map(_.toFloat).toArray
    b = f(1).split(",").map(_.toFloat).toArray
  }
}

