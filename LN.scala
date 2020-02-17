package layer
import breeze.linalg._
import pll.utils.RichArray._
class LN(val xn:Int)extends Layer{
  override type T = Float
  val ep = 0.00000001f
  var beta = 0f
  var gamma = 1f
  var sbeta = 0f
  var sgamma = 0f
  var rbeta = 0f
  var rgamma = 0f
  var p1t = 1f //updataで使うp1のt乗
  var p2t = 1f
  var dbeta = 0f
  var dgamma = 0f
  var myu = 0f
  var sig = 0f
  var xHat = new Array[T](xn)
  var ivar = 0f
  var sqrtvar = 0f
  var xmyu = new Array[T](xn)
  def forward(x:Array[T])={
    var y = new Array[T](xn)
    myu = 0f
    sig = 0f
    //--------myu-----------
    for(i<-0 until xn){
      myu += x(i) / xn
    }
    //--------sig----------
    for(i<-0 until xn){
      sig += (x(i)-myu)*(x(i)-myu) / xn
    }
    //--------xHat-----------
    for(i<-0 until xn){
      xHat(i) = (x(i)-myu)/math.sqrt(sig+ep).toFloat
    }
    //---------y-----------
    for(i<-0 until xn){
      y(i) = gamma*xHat(i) + beta
    }
    //------ivar---------
    for(i<-0 until xn){
      ivar = 1f/math.sqrt(sig+ep).toFloat
    }
    //---------sqrtvar------------
    for(i<-0 until xn){
      sqrtvar = math.sqrt(sig+ep).toFloat
    }
    //---------xmyu--------------
    for(i<-0 until xn){
      xmyu(i) = x(i) - myu
    }
    y
  }
  override def forward(xs:Array[Array[T]])={
    xs.map(forward)
  }
  def backward(ds:Array[T])={
    var dxHat = new Array[T](xn)
    var divar = 0f
    var dxmyu1 = new Array[T](xn)
    var dsqrtvar = 0f
    var dvar = 0f
    var dsq = new Array[T](xn)
    var dxmyu2 = new Array[T](xn)
    var dmyu = 0f
    var dx1 = new Array[T](xn)
    var dx2 = new Array[T](xn)
    var dx = new Array[T](xn)
    dgamma = 0f
    dbeta = 0f
    divar = 0f
    dmyu = 0f
    //------------dΓ--------------------
    for(i<-0 until xn){
      dgamma += ds(i)*xHat(i)
    }
    //-------------dβ-------------------
    for(i<-0 until xn){
      dbeta += ds(i)
    }
    //------------d1---------------------
    for(i<-0 until xn){
      dxHat(i) = ds(i) * gamma
    }
    //-----------d2------------------------
    for(i<-0 until xn){
      divar += dxHat(i) * xmyu(i)
    }
    //--------------d3----------------------------
    for(i<-0 until xn){
      dsqrtvar = - divar / (sqrtvar*sqrtvar)
    }
    //--------------d4---------------------------
    for(i<-0 until xn){
      dvar = dsqrtvar / (math.sqrt(sig+ep).toFloat*2f)
    }
    //--------------d5------------------------
    for(i<-0 until xn){
      dsq(i) = dvar / xn
    }
    //--------------d6---------------------
    for(i<-0 until xn){
      dxmyu2(i) = 2f*xmyu(i)*dsq(i)
    }
    //--------------d7----------------------
    for(i<-0 until xn){
      dxmyu1(i) = dxHat(i) * ivar
    }
    //---------------d8---------------------
    for(i<-0 until xn){
      dmyu += -(dxmyu1(i)+dxmyu2(i))
    }
    //--------------d9----------------------
    for(i<-0 until xn){
      dx2(i) = dmyu / xn
    }
    //-------------d10----------------------
    for(i<-0 until xn){
      dx1(i) = dxmyu1(i)+dxmyu2(i)
    }
    //---------------dx----------------------
    for(i<-0 until xn){
      dx(i) = dx1(i) + dx2(i)
    }
    dx
  }
  override def backward(ds:Array[Array[T]])={
    ds.reverse.map(backward).reverse
  }
  def update(){
    //betaとgammaを更新
    //val ep = 0.001
    val ep = 0.002f //GANver.
    val p1 = 0.5f //GANver.
                  //val p1 = 0.9
    val p2 = 0.999f
    val delta = 0.00000001f
    p1t = p1t * p1 //updata（p1）が何回呼び出されたか
    p2t = p2t * p2
    sbeta = p1 * sbeta + (1f-p1) * dbeta
    sgamma = p1 * sgamma + (1f-p1) * dgamma
    rbeta = p2 * rbeta + (1f-p2) * ( dbeta *:* dbeta )
    rgamma = p2 * rgamma + (1f-p2) * ( dgamma *:* dgamma )
    val s_beta = sbeta / (1f-p1t)
    val s_gamma = sgamma / (1f-p1t)
    val r_beta = rbeta / (1f-p2t)
    val r_gamma = rgamma / (1f-p2t)
    beta += -ep * s_beta / (math.sqrt(r_beta).toFloat + delta )
    gamma += -ep * s_gamma / (math.sqrt(r_gamma).toFloat + delta )
    reset()
  }
  def reset(){
    dbeta = 0f
    dgamma = 0f
  }
}
