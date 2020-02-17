package layer
import breeze.linalg._
import pll.utils.RichArray._
class BN(val xn:Int)extends Layer{
  override type T = Float
  val ep = 0.00000001f
  var beta = DenseVector.zeros[T](xn)
  var gamma = DenseVector.ones[T](xn)
  var sbeta = DenseVector.zeros[T](xn)
  var sgamma = DenseVector.ones[T](xn)
  var rbeta = DenseVector.zeros[T](xn)
  var rgamma = DenseVector.ones[T](xn)
  var p1t = 1f //updataで使うp1のt乗
  var p2t = 1f
  var dbeta = DenseVector.zeros[T](xn)
  var dgamma = DenseVector.ones[T](xn)
  var myu = DenseVector.zeros[T](xn)
  var sig = DenseVector.zeros[T](xn)
  var xHat = Array.ofDim[T](xn,xn)
  var ivar = DenseVector.zeros[T](xn)
  var sqrtvar = DenseVector.zeros[T](xn)
  var xmyu = Array.ofDim[T](xn,xn)
  override def forward(xs:Array[T]):Array[T]={
    xs
  }
  override def forward(xs:Array[Array[T]]):Array[Array[T]]={
    var y = Array.ofDim[T](xs.size,xn)
    myu = DenseVector.zeros[T](xn)
    sig = DenseVector.zeros[T](xn)
    //--------myu-----------
    for(i<-0 until xs.size){
      for(j<-0 until xn){
        myu(j) += xs(i)(j) / xs.size
      }
    }
    //--------sig----------
    for(i<-0 until xs.size){
      for(j<-0 until xn){
        sig(j) += (xs(i)(j)-myu(j))*(xs(i)(j)-myu(j)) / xs.size
      }
    }
    xHat = Array.ofDim[T](xs.size,xn)
    //--------xHat-----------
    for(i<-0 until xs.size){
      for(j<-0 until xn){
        xHat(i)(j) = (xs(i)(j)-myu(j))/math.sqrt(sig(j)+ep).toFloat
      }
    }
    //---------y-----------
    for(i<-0 until xs.size){
      for(j<-0 until xn){
        y(i)(j) = gamma(j)*xHat(i)(j) + beta(j)
      }
    }
    //------ivar---------
    for(i<-0 until xs.size ; j<-0 until xn){
      ivar(j) = 1f/math.sqrt(sig(j)+ep).toFloat
    }
    //---------sqrtvar------------
    for(i<-0 until xs.size ; j<-0 until xn){
      sqrtvar(j) = math.sqrt(sig(j)+ep).toFloat
    }
    xmyu = Array.ofDim[T](xs.size,xn)
    //---------xmyu--------------
    for(i<-0 until xs.size ; j<-0 until xn){
      xmyu(i)(j) = xs(i)(j) - myu(j)
    }
    y
  }
  override def backward(ds:Array[T]):Array[T]={
    ds
  }
  override def backward(ds:Array[Array[T]])={
    var dxHat = Array.ofDim[T](ds.size,xn)
    var divar = DenseVector.zeros[T](xn)
    var dxmyu1 = Array.ofDim[T](ds.size,xn)
    var dsqrtvar = DenseVector.zeros[T](xn)
    var dvar = DenseVector.zeros[T](xn)
    var dsq = Array.ofDim[T](ds.size,xn)
    var dxmyu2 = Array.ofDim[T](ds.size,xn)
    var dmyu = DenseVector.zeros[T](xn)
    var dx1 = Array.ofDim[T](ds.size,xn)
    var dx2 = Array.ofDim[T](ds.size,xn)
    var dx = Array.ofDim[T](ds.size,xn)
    dgamma = DenseVector.zeros[T](xn)
    dbeta = DenseVector.zeros[T](xn)
    dmyu = DenseVector.zeros[T](xn)
    divar = DenseVector.zeros[T](xn)
    //------------dΓ--------------------
    for(i<-0 until ds.size ; j<-0 until xn){
      dgamma(j) += ds(i)(j)*xHat(i)(j)
    }
    //-------------dβ-------------------
    for(i<-0 until ds.size ; j<-0 until xn){
      dbeta(j) += ds(i)(j)
    }
    //------------d1---------------------
    for(i<-0 until ds.size ; j<-0 until xn){
      dxHat(i)(j) = ds(i)(j) * gamma(j)
    }
    //-----------d2------------------------
    for(i<-0 until ds.size ; j<-0 until xn){
      divar(j) += dxHat(i)(j) * xmyu(i)(j)
    }
    //--------------d3----------------------------
    for(j<-0 until xn){
      dsqrtvar(j) = - divar(j) / (sqrtvar(j)*sqrtvar(j))
    }
    //--------------d4---------------------------
    for(i<-0 until ds.size ; j<-0 until xn){
      dvar(j) = dsqrtvar(j) / (math.sqrt(sig(j)+ep).toFloat*2)
    }
    //--------------d5------------------------
    for(i<-0 until ds.size ; j<-0 until xn){
      dsq(i)(j) = dvar(j) / ds.size
    }
    //--------------d6---------------------
    for(i<-0 until ds.size ; j<-0 until xn){
      dxmyu2(i)(j) = 2f*xmyu(i)(j)*dsq(i)(j)
    }
    //--------------d7----------------------
    for(i<-0 until ds.size ; j<-0 until xn){
      dxmyu1(i)(j) = dxHat(i)(j) * ivar(j)
    }
    //---------------d8---------------------
    for(i<-0 until ds.size ; j<-0 until xn){
      dmyu(j) += -(dxmyu1(i)(j)+dxmyu2(i)(j))
    }
    //--------------d9----------------------
    for(i<-0 until ds.size ; j<-0 until xn){
      dx2(i)(j) = dmyu(j) / ds.size
    }
    //-------------d10----------------------
    for(i<-0 until ds.size ; j<-0 until xn){
      dx1(i)(j) = dxmyu1(i)(j)+dxmyu2(i)(j)
    }
    //---------------dx----------------------
    for(i<-0 until ds.size ; j<-0 until xn){
      dx(i)(j) = dx1(i)(j) + dx2(i)(j)
    }
    dx
  }
  def update(){
    //betaとgammaを更新
    val ep = 0.002f //GANver.
    val p1 = 0.5f //GANver.
                  //val ep = 0.001
                  //val p1 = 0.9
    val p2 = 0.999f
    val delta = 0.00000001f
    p1t = p1t * p1 //updata（p1）が何回呼び出されたか
    p2t = p2t * p2
    sbeta = p1 * sbeta + (1f-p1) * dbeta
    sgamma = p1 * sgamma + (1f-p1) * dgamma
    rbeta = p2 * rbeta + (1f-p2) * ( dbeta *:* dbeta )
    rgamma = p2 * rgamma + (1f-p2) * ( dgamma *:* dgamma )
    val s_beta = sbeta /:/ (1f-p1t)
    val s_gamma = sgamma /:/ (1f-p1t)
    val r_beta = rbeta /:/ (1f-p2t)
    val r_gamma = rgamma /:/ (1f-p2t)
    beta += -ep * s_beta /:/ (r_beta.map(a=>math.sqrt(a).toFloat) + delta )
    gamma += -ep * s_gamma /:/ (r_gamma.map(a=>math.sqrt(a).toFloat) + delta )
    //reset()
  }
  def reset(){
    dbeta = DenseVector.zeros[T](xn)
    dgamma = DenseVector.zeros[T](xn)
  }
}
