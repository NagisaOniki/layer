package layer
import breeze.linalg._
import pll.utils.RichArray._
object Adam {
  type T = Float
  class Adam_BNa(val n:Int, val eps:T = 0.002f, val rho1:T  = 0.5f, val rho2:T  = 0.999f) {
    val delta = 1e-5f
    var rho1t = 1f
    var rho2t = 1f
    var s = Array.ofDim[T](n)
    var r = Array.ofDim[T](n)
    def update(K:Array[T], dK:Array[T]) = {
      var nK = Array.ofDim[T](K.size)
      rho1t *= rho1
      rho2t *= rho2
      val rho1tr = 1f / (1f - rho1t)
      val rho2tr = 1f / (1f - rho2t)
      for(i <- 0 until K.size) {
        s(i) = rho1 * s(i) + (1f - rho1) * dK(i)
        r(i) = rho2 * r(i) + (1f - rho2) * dK(i) * dK(i)
        val d = (s(i) * rho1tr) / (math.sqrt(r(i) * rho2tr).toFloat + delta)
        K(i) = K(i) - eps * d
      }
    }
  }
  class Adam_D(val n:Int,val eps:T=0.001f) {
    val delta = (1e-8).toFloat
    val rho1 = 0.5f
    val rho2 = 0.999f
    var rho1t = 1f
    var rho2t = 1f
    var s = Array.ofDim[T](n)
    var r = Array.ofDim[T](n)
    def update(K:Array[T], dK:Array[T],count:T) = {
      rho1t *= rho1
      rho2t *= rho2
      val rho1tr = 1f / (1f - rho1t)
      val rho2tr = 1f / (1f - rho2t)
      for(i <- 0 until K.size) {
        s(i) = rho1 * s(i) + (1f - rho1) * dK(i)
        r(i) = rho2 * r(i) + (1f - rho2) * dK(i) * dK(i)
        val d = (s(i) * rho1tr) / (math.sqrt(r(i) * rho2tr).toFloat + delta)
        K(i) = K(i) - eps/count * d
      }
    }
  }
  class Adam_DM(val rows:Int, val cols:Int) {
    val eps = 0.001f
    val delta = (1e-8).toFloat
    val rho1 = 0.9f
    val rho2 = 0.999f
    var rho1t = 1f
    var rho2t = 1f
    var s = DenseMatrix.zeros[T](rows,cols)
    var r = DenseMatrix.zeros[T](rows,cols)
    def update(K:DenseMatrix[T], dK:DenseMatrix[T],count:T) = {
      rho1t *= rho1
      rho2t *= rho2
      val rho1tr = 1f / (1f - rho1t)
      val rho2tr = 1f / (1f - rho2t)
      for(i <- 0 until K.rows; j <- 0 until K.cols) {
        s(i,j) = rho1 * s(i,j) + (1f - rho1) * dK(i,j)
        r(i,j) = rho2 * r(i,j) + (1f - rho2) * dK(i,j) * dK(i,j)
        val d = (s(i,j) * rho1tr) / (math.sqrt(r(i,j) * rho2tr).toFloat + delta)
        K(i,j) = K(i,j) - eps/count * d
      }
    }
  }
  class Adam_DV(val n:Int) {
    val eps = 0.001f
    val delta = 1e-8f
    val rho1 = 0.9f
    val rho2 = 0.999f
    var rho1t = 1f
    var rho2t = 1f
    var s = DenseVector.zeros[T](n)
    var r = DenseVector.zeros[T](n)
    def update(K:DenseVector[T], dK:DenseVector[T],count:T) = {
      rho1t *= rho1
      rho2t *= rho2
      val rho1tr = 1f / (1f - rho1t)
      val rho2tr = 1f / (1f - rho2t)
      for(i <- 0 until K.size) {
        s(i) = rho1 * s(i) + (1f - rho1) * dK(i)
        r(i) = rho2 * r(i) + (1f - rho2) * dK(i) * dK(i)
        val d = (s(i) * rho1tr) / (math.sqrt(r(i) * rho2tr).toFloat + delta)
        K(i) = K(i) - eps/count * d
      }
    }
  }
}
