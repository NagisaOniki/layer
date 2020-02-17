package layer
import layer.Adam._
import breeze.linalg._
import pll.utils.RichArray._
class BNa(val xn:Int, val eps:T = 0.001f, val rho1:T = 0.9f, val rho2:T = 0.999f) extends Layer {
  override type T = Float
  var gamma = new Array[T](xn).map(_ => 1 : T)
  var beta = new Array[T](xn)
  var dgamma = new Array[T](gamma.size)
  var dbeta = new Array[T](beta.size)
  val adam_gamma = new Adam_BNa(gamma.size,eps,rho1,rho2)
  val adam_beta = new Adam_BNa(beta.size,eps,rho1,rho2)
  var xmu = Array.ofDim[T](1,xn) // rhs is just a placeholder value
  var sigma = new Array[T](xn)
  val delta = 1e-5f
  var mmu = new Array[T](xn)
  var msigma = new Array[T](xn)
  val decay = 0.999f
  def forward(x:Array[T]) : Array[T] = {
    val y = new Array[T](xn)
    for(i <- 0 until xn) {
      val xh = (x(i) - mmu(i)) / (msigma(i) + delta)
      y(i) = xh * gamma(i) + beta(i)
    }
    y
  }
  def backward(d:Array[T]) : Array[T] = {
    d
  }
  override def forward(xs:Array[Array[T]]) : Array[Array[T]] = {
    val m = xs.size
    xmu = Array.ofDim[T](m,xn)
    for(j <- 0 until xn) {
      var mu = 0f
      for(i <- 0 until m) {
        mu += xs(i)(j)
      }
      mu /= m
      mmu(j) = decay * mmu(j) + (1f-decay) * mu
      for(i <- 0 until m) {
        xmu(i)(j) = xs(i)(j) - mu
        sigma(j) += xmu(i)(j) * xmu(i)(j)
      }
      sigma(j) = math.sqrt(sigma(j) / m + delta).toFloat
      msigma(j) = decay * msigma(j) + (1f-decay) * sigma(j)
    }
    var ys = Array.ofDim[T](m,xn)
    for(j <- 0 until xn) {
      for(i <- 0 until m) {
        ys(i)(j) = gamma(j) * xmu(i)(j) / sigma(j) + beta(j)
      }
    }
    ys
  }
  override def backward(ds:Array[Array[T]]) : Array[Array[T]] = {
    val m = ds.size
    var dx = Array.ofDim[T](m,xn)
    for(j <- 0 until xn) {
      for(i <- 0 until m) {
        dbeta(j) += ds(i)(j)
        dgamma(j) += ds(i)(j) * xmu(i)(j) / sigma(j)
      }
      var d1 = new Array[T](m)
      var d2 = 0f
      for(i <- 0 until m) {
        d1(i) = gamma(j) * ds(i)(j)
        d2 += xmu(i)(j) * d1(i)
      }
      val d3 = -d2 / (sigma(j) * sigma(j))
      val d4 = d3 / (2f *  sigma(j))
      var d8 = 0f
      var d10 = new Array[T](m)
      for(i <- 0 until m) {
        val d5 = d4 / m
        val d6 = 2f * xmu(i)(j) * d5
        val d7 = d1(i) / sigma(j)
        d10(i) = d6 + d7
        d8 -= d10(i)
      }
      val d9 = d8 / m
      for(i <- 0 until m) {
        dx(i)(j) = d9 + d10(i)
      }
    }
    dx
  }
  def update() {
    adam_beta.update(beta,dbeta)
    adam_gamma.update(gamma,dgamma)
    reset()
  }
  def reset() {
    dgamma = new Array[T](gamma.size)
    dbeta = new Array[T](beta.size)
  }
}
