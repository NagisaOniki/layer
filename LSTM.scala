package layer
import breeze.linalg._
import pll.utils.RichArray._
class LSTM(val xn:Int , val yn:Int)extends Layer{
  override type T = Float
  val rand = new scala.util.Random(0)
  var bi = DenseVector.zeros[T](yn)
  var bf = DenseVector.zeros[T](yn)
  var bo = DenseVector.zeros[T](yn)
  var bh_ = DenseVector.zeros[T](yn)
  var Wir = DenseMatrix.zeros[T](yn,xn).map(_=>rand.nextGaussian.toFloat*0.01f)
  var Wfr = DenseMatrix.zeros[T](yn,xn).map(_=>rand.nextGaussian.toFloat*0.01f)
  var Wor = DenseMatrix.zeros[T](yn,xn).map(_=>rand.nextGaussian.toFloat*0.01f)
  var Wh_r = DenseMatrix.zeros[T](yn,xn).map(_=>rand.nextGaussian.toFloat*0.01f)
  var Wix = DenseMatrix.zeros[T](yn,xn).map(_=>rand.nextGaussian.toFloat*0.01f)
  var Wfx = DenseMatrix.zeros[T](yn,xn).map(_=>rand.nextGaussian.toFloat*0.01f)
  var Wox = DenseMatrix.zeros[T](yn,xn).map(_=>rand.nextGaussian.toFloat*0.01f)
  var Wh_x = DenseMatrix.zeros[T](yn,xn).map(_=>rand.nextGaussian.toFloat*0.01f)

  var dC = DenseVector.zeros[T](yn)
  var dr = DenseVector.zeros[T](yn)
  var dx = DenseVector.zeros[T](yn)

  var dbi = DenseVector.zeros[T](yn)
  var dbf = DenseVector.zeros[T](yn)
  var dbo = DenseVector.zeros[T](yn)
  var dbh_ = DenseVector.zeros[T](yn)
  var dWir = DenseMatrix.zeros[T](yn,xn)
  var dWfr = DenseMatrix.zeros[T](yn,xn)
  var dWor = DenseMatrix.zeros[T](yn,xn)
  var dWh_r = DenseMatrix.zeros[T](yn,xn)
  var dWix = DenseMatrix.zeros[T](yn,xn)
  var dWfx = DenseMatrix.zeros[T](yn,xn)
  var dWox = DenseMatrix.zeros[T](yn,xn)
  var dWh_x = DenseMatrix.zeros[T](yn,xn)

  var is = List[DenseVector[T]](DenseVector.zeros[T](xn))
  var h_s =List[DenseVector[T]](DenseVector.zeros[T](xn))
  var fs = List[DenseVector[T]](DenseVector.zeros[T](xn))
  var os = List[DenseVector[T]](DenseVector.zeros[T](xn))
  var xs = List[DenseVector[T]](DenseVector.zeros[T](xn))
  var cs = List[DenseVector[T]](DenseVector.zeros[T](yn)) //記憶のリスト
  var hs = List[DenseVector[T]](DenseVector.zeros[T](yn)) //出力のリスト
                                                          //Adam
  var p1t = 1f
  var p2t = 1f
  var sbi = DenseVector.zeros[T](yn)
  var sbf = DenseVector.zeros[T](yn)
  var sbo = DenseVector.zeros[T](yn)
  var sbh_ = DenseVector.zeros[T](yn)
  var sWir = DenseMatrix.zeros[T](yn,xn)
  var sWfr = DenseMatrix.zeros[T](yn,xn)
  var sWor = DenseMatrix.zeros[T](yn,xn)
  var sWh_r = DenseMatrix.zeros[T](yn,xn)
  var sWix = DenseMatrix.zeros[T](yn,xn)
  var sWfx = DenseMatrix.zeros[T](yn,xn)
  var sWox = DenseMatrix.zeros[T](yn,xn)
  var sWh_x = DenseMatrix.zeros[T](yn,xn)

  var rbi = DenseVector.zeros[T](yn)
  var rbf = DenseVector.zeros[T](yn)
  var rbo = DenseVector.zeros[T](yn)
  var rbh_ = DenseVector.zeros[T](yn)
  var rWir = DenseMatrix.zeros[T](yn,xn)
  var rWfr = DenseMatrix.zeros[T](yn,xn)
  var rWor = DenseMatrix.zeros[T](yn,xn)
  var rWh_r = DenseMatrix.zeros[T](yn,xn)
  var rWix = DenseMatrix.zeros[T](yn,xn)
  var rWfx = DenseMatrix.zeros[T](yn,xn)
  var rWox = DenseMatrix.zeros[T](yn,xn)
  var rWh_x = DenseMatrix.zeros[T](yn,xn)

  def push(x:DenseVector[T]) = { xs ::= x; x }
  def pop() = { val x = xs.head; xs = xs.tail; x }
  def forward(x0:Array[T]) : Array[T] ={
    val x = DenseVector(x0)
    push(x)
    val hprev = hs.head
    val Cprev = cs.head
    val i = (Wix * x + Wir * hprev + bi).map(a=>sigmoid(a).toFloat)
    val h_ = (Wh_x * x + Wh_r * hprev + bh_).map(a=>tanh(a).toFloat)
    val f = (Wfx * x + Wfr * hprev + bf).map(a=>sigmoid(a).toFloat)
    val o = (Wox * x + Wor * hprev + bo).map(a=>sigmoid(a).toFloat)
    val C = i *:* h_ + f *:* Cprev
    val h = o * C.map(a=>tanh(a).toFloat)
    is = i :: is
    h_s = h_ :: h_s
    fs = f :: fs
    os = o :: os
    xs = x :: xs
    cs = C :: cs
    hs = h :: hs
    h.toArray
  }
  override def forward(x:Array[Array[T]])={
    x.map(forward)
  }
  def backward(d:Array[T])={ //引数はdはdhのこと
    val x0 = pop()
    val dh = DenseVector(d)
    val Z = cs.head.map(a=>tanh(a).toFloat)
    val A = os.head *:* (dh + dr) *:* (1f - Z *:* Z)
    val dbit = ( dC + A ) *:* h_s.head * is.head * ( 1f - is.head )
    val dbft = ( dC + A ) * cs.tail.head * fs.head * ( 1f - fs.head )
    val dbot = ( dh + dr ) * cs.head.map(a=>tanh(a).toFloat) * os.head * ( 1f - os.head )
    val dbh_t = ( dC + A ) * is.head * ( 1f - h_s.head * h_s.head )
    //dC,dr,dxを更新
    dC = ( dC + A ) *:* fs.head
    dr = Wor.t * dbot + Wfr.t * dbft +  Wh_r.t * dbh_t + Wir.t * dbit
    dx = Wox.t * dbot + Wfx.t * dbft +  Wix.t * dbh_t + Wix.t * dbit
    dbi += dbit
    dbf += dbft
    dbo += dbot
    dbh_ += dbh_t
    val dWirt = dbit * hs.tail.head.t
    val dWfrt = dbft * hs.tail.head.t
    val dWort = dbot * hs.tail.head.t
    val dWh_rt = dbh_t * hs.tail.head.t
    dWir += dWirt
    dWfr += dWfrt
    dWor += dWort
    dWh_r += dWh_rt
    val dWixt = dbit * xs.head.t
    val dWfxt = dbft * xs.head.t
    val dWoxt = dbot * xs.head.t
    val dWh_xt = dbh_t * xs.head.t
    dWix += dWixt
    dWfx += dWfxt
    dWox += dWoxt
    dWh_x += dWh_xt
    is = is.tail
    h_s = h_s.tail
    fs = fs.tail
    os = os.tail
    xs = xs.tail
    cs = cs.tail
    hs = hs.tail
    dx.toArray
  }
  override def backward(d:Array[Array[T]])={
    d.reverse.map(backward).reverse
  }
  def Adam0(a0:DenseVector[T],da0:DenseVector[T],sa0:DenseVector[T],ra0:DenseVector[T])={
    var a = a0
    var da = da0
    var sa = sa0
    var ra = ra0
    val ep = 0.001f //学習率
    val p1 = 0.9f
    val p2 = 0.999f
    val delta = 0.00000001f
    p1t = p1t*p1
    p2t = p2t*p2
    sa = p1 * sa + (1f-p1) * da
    ra = p2 * ra + (1f-p2) * (da *:* da)
    val s_a = sa /:/ (1f-p1t)
    val r_a = ra /:/ (1f-p2t)
    a += -ep * s_a / (r_a.map(a=>math.sqrt(a).toFloat)+delta)
    a
  }
  def Adam1(a0:DenseMatrix[T],da0:DenseMatrix[T],sa0:DenseMatrix[T],ra0:DenseMatrix[T])={
    var a = a0
    var da = da0
    var sa = sa0
    var ra = ra0
    val ep = 0.001f //学習率
    val p1 = 0.9f
    val p2 = 0.999f
    val delta = 0.00000001f
    p1t = p1t*p1
    p2t = p2t*p2
    sa = p1 * sa + (1f-p1) * da
    ra = p2 * ra + (1f-p2) * (da *:* da)
    val s_a = sa /:/ (1f-p1t)
    val r_a = ra /:/ (1f-p2t)
    a += -ep * s_a / (r_a.map(a=>math.sqrt(a).toFloat)+delta)
    a
  }
  def update(){
    var layer1 = Array(bi,bf,bo,bh_)
    var layer1_ = Array(Wir,Wfr,Wor,Wh_r,Wix,Wfx,Wox,Wh_x)
    val layer2 = Array(dbi,dbf,dbo,dbh_)
    val layer2_ = Array(dWir,dWfr,dWor,dWh_r,dWix,dWfx,dWox,dWh_x)
    val layer3 = Array(sbi,sbf,sbo,sbh_)
    val layer3_ = Array(sWir,sWfr,sWor,sWh_r,sWix,sWfx,sWox,sWh_x)
    val layer4 = Array(rbi,rbf,rbo,rbh_)
    val layer4_ = Array(rWir,rWfr,rWor,rWh_r,rWix,rWfx,rWox,rWh_x)
    for(i<-0 until 4){
      layer1(i) = Adam0(layer1(i),layer2(i),layer3(i),layer4(i))
    }
    for(i<-0 until 8){
      layer1_(i) = Adam1(layer1_(i),layer2_(i),layer3_(i),layer4_(i))
    }
    reset()
  }
  def setstate(beforeh:DenseVector[T]){
    hs ::= beforeh
  }
  def reset(){
    dbi = DenseVector.zeros[T](yn)
    dbf = DenseVector.zeros[T](yn)
    dbo = DenseVector.zeros[T](yn)
    dbh_ = DenseVector.zeros[T](yn)

    dWir = DenseMatrix.zeros[T](yn,xn)
    dWfr = DenseMatrix.zeros[T](yn,xn)
    dWor = DenseMatrix.zeros[T](yn,xn)
    dWh_r = DenseMatrix.zeros[T](yn,xn)

    dWix = DenseMatrix.zeros[T](yn,xn)
    dWfx = DenseMatrix.zeros[T](yn,xn)
    dWox = DenseMatrix.zeros[T](yn,xn)
    dWh_x = DenseMatrix.zeros[T](yn,xn)

    dr = DenseVector.zeros[T](yn)
    dC = DenseVector.zeros[T](yn)
  }
  def tanh(x:T)={
    (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
  }
  def sigmoid(u:T)={
    1f/(1f+math.exp(-u))
  }
}
