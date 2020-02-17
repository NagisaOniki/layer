package layer
import breeze.linalg._
import pll.utils.RichArray._
class Affine(val xn:Int, val yn:Int) extends Layer {
  override type T = Float
  val rand = new util.Random(0)
  var W = DenseMatrix.zeros[T](yn,xn).map(_=>rand.nextGaussian.toFloat *0.01f) // > /math.sqrt(xn).toFloat (精度比較)
  var b = DenseVector.zeros[T](yn)
  var dW = DenseMatrix.zeros[T](yn,xn)
  var db = DenseVector.zeros[T](yn)
  var xList = List[Array[T]]()
  var sW = DenseMatrix.zeros[T](yn,xn)
  var sb = DenseVector.zeros[T](yn)
  var rW = DenseMatrix.zeros[T](yn,xn)
  var rb = DenseVector.zeros[T](yn)
  var p1t = 1f //updataで使うp1のt乗
  var p2t = 1f
  var bc = 0f //backwardCount
  def push(x:Array[T]) = { xList ::= x; x }
  def pop() = { val x = xList.head; xList = xList.tail; x }
  def forward(x:Array[T]) = {
    push(x)
    val xv = DenseVector(x)
    val y = W * xv + b
    y.toArray
  }
  /* override def forward(xs:Array[Array[T]]) = {
   xs.map(push)
   val result_matrix = Array.ofDim[T](W.rows * xs.size)
   val left_matrix = DenseMatrix.horzcat(W , DenseMatrix(b).t).t.toArray
   val right_matrix = new DenseMatrix(xs(0).size+1 , xs.size ,xs.map(_ :+ 1f).flatten).t.toArray //行、列、配列
   blas.BLAS.matmulF(left_matrix,right_matrix,result_matrix,W.rows,xs.size,W.cols+1)
   new DenseMatrix(xs.size , W.rows ,  result_matrix).t.toArray.grouped(yn).toArray
   }*/
  override def forward(xs:Array[Array[T]])={
    xs.map(forward)
  }
  /*def forward(xs:Array[Array[T]]) = { //nakajiForward
   xs.map(push)
   // call blas library
   val wx= Array.ofDim[T](W.rows*xs.size)
   blas.BLAS.matmulF(W.toArray, xs.flatten.transpose(xs.size, W.cols), wx, W.rows, xs.size, W.cols)

   wx.transpose(W.rows, xs.size).grouped(W.rows).map(_ + b.toArray).toArray
   }*/
  def backward(d:Array[T]) = {
    bc += 1f
    val x = pop()
    val dv = DenseVector(d)
    // dW,dbを計算する ★
    dW += dv * DenseVector(x).t
    db += dv
    var dx = DenseVector.zeros[T](xn)
    // dxを計算する ★
    dx = W.t * dv
    dx.toArray
  }
  override def backward(ds:Array[Array[T]])={
    ds.reverse.map(backward).reverse
  }
  /*override def backward(ds:Array[Array[T]])={
   bc += 1f
   var xs = Array.ofDim[T](ds.size,xn)
   for(i<-0 until ds.size){
   xs(i) = pop()
   }
   //dW,db
   val result_matrix = new Array[T](yn * xn + yn)
   val left_matrix = new DenseMatrix(yn , ds.size , ds.flatten).t.toArray
   val right_matrix = (xs.map(_:+1f)).reverse.flatten
   blas.BLAS.matmulF(left_matrix , right_matrix , result_matrix  , yn , xn+1 , ds.size)
   var tempdW = new Array[T](yn*xn)
   var tempdb = new Array[T](yn)
   for(i<-0 until yn){
   for(j<-0 until xn+1){
   if(j<xn){
   tempdW(i*xn+j) = result_matrix(i*(xn+1)+j)
   }else{
   tempdb(i) = result_matrix(i*(xn+1)+j)
   }
   }
   }
   dW = DenseMatrix(tempdW)
   db = DenseVector(tempdb)
   //dx
   val result_matrix2 = new Array[T](xn * ds.size)//dxs
   val left_matrix2 = W.toArray
   val right_matrix2 = new DenseMatrix(yn , ds.size , ds.flatten).t.toArray
   blas.BLAS.matmulF(left_matrix2 , right_matrix2 , result_matrix2 , xn , ds.size , yn)
   new DenseMatrix(xn , ds.size ,  result_matrix2).toArray.grouped(yn).toArray
   }*/
  /*def backward(ds:Array[Array[Float]]) = { //nakajiBackward
   val dloss = ds.flatten.transpose(ds.size, W.rows)
   val dbias = ds.reduce(_ + _)
   // call blas library
   val past_xs = xs.take(ds.size).reverse
   xs = xs.drop(ds.size)
   val dweight = Array.ofDim[Float](W.size)
   blas.BLAS.matmul(dloss, past_xs.toArray.flatten, dweight, W.rows, W.cols, ds.size)
   // memory grad
   Range(0, b.size).foreach(i => db(i) += dbias(i))
   Range(0, W.size).foreach(i => dW(i) += dweight(i))
   // call blas library
   val dinput = Array.ofDim[Float](W.cols*ds.size)
   blas.BLAS.matmul(W.transpose(W.rows, W.cols), dloss, dinput, W.cols, ds.size, W.rows)
   dinput.transpose(W.cols, ds.size)
   .grouped(W.cols)
   .toArray
   }*/
  def update() {
    //GAN ver.
    val ep = 0.01f/bc
    bc = 0f
    val p1 = 0.5f
    /*      //normal ver.
     val ep = 0.1f //学習率
     val p1 = 0.9f //モーメントの推定に対する指数減衰率*/
    val p2 = 0.999f //モーメントの推定に対する指数減衰率
    val delta = 0.00000001f //数値安定のための小さな定数デルタ
    p1t = p1t * p1 //updata（p1）が何回呼び出されたか
    p2t = p2t * p2
    sW = p1 * sW + (1f-p1) * dW
    sb = p1 * sb + (1f-p1) * db
    rW = p2 * rW + (1f-p2) * ( dW *:* dW )
    rb = p2 * rb + (1f-p2) * ( db *:* db )
    val s_W = sW /:/ (1f-p1t)
    val s_b = sb /:/ (1f-p1t)
    val r_W = rW /:/ (1f-p2t)
    val r_b = rb /:/ (1f-p2t)
    W += -ep * s_W / (r_W.map(a=>math.sqrt(a).toFloat) + delta ) //+ W.map(a=>math.abs(a).toFloat)
      b += -ep * s_b / (r_b.map(a=>math.sqrt(a).toFloat) + delta ) //+ b.map(a=>math.abs(a).toFloat)
     // println("W:"+W(0,0))
   // println("dW:"+dW(0,0))
   // println("db:"+db(0))
      //reset()
  }
  override def save(fn:String){
    val save = new java.io.PrintWriter(fn)
    for(i<-0 until yn ; j<-0 until xn){
      save.println(W(i,j))
    }
    for(i<-0 until yn){
      save.println(b(i))
    }
    save.close
  }
  override def load(fn:String){
    val f = scala.io.Source.fromFile(fn).getLines.toArray
    for(i<-0 until yn ; j<-0 until xn){
      W(i,j) = f(i*xn + j).toFloat
    }
    for(i<-0 until yn){
      b(i) = f(yn*xn + i).toFloat
    }
  }
  def reset() {
    dW = DenseMatrix.zeros[T](yn,xn)
    db = DenseVector.zeros[T](yn)
    xList = List[Array[T]]()
  }
}
