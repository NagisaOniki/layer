/*package layer
import layers._
import breeze.linalg._
import java.io.{FileOutputStream => FileStream, OutputStreamWriter => StreamWriter}
import updater._

object test_c {
  def test() = {
    val C = new FConvolution(2,3,3,3,2)
    C.K = Array(1f,1f,1f,1f,2f,2f,2f,2f,3f,3f,3f,3f,4f,4f,4f,4f,5f,5f,5f,5f,6f,6f,6f,6f)
    val xs = Array(Array(1f,2f,3f,4f,5f,6f,7f,8f,9f,2f,3f,4f,5f,6f,7f,8f,9f,10f,3f,4f,5f,6f,7f,8f,9f,10f,11f))

    val y = C.forward(xs)
    C.backward(Array(Array(1f,1f,1f,1f,2f,2f,2f,2f)))
    val d = C.dk


    for(i <- y(0)){
    //  print(i + ",")
    }
    for(j <- d){
      print(j + ",")
    }
  }
}


class FConvolution(val KW:Int,val IH:Int,val IW:Int,val IC:Int,val OC:Int , val e:Float = 0.01f , val rhot1:Float = 0.9f ,val rhot2:Float = 0.999f ,val c:Float = 0f,val st:Int = 1 ,val af:Float = 0.01f) extends Layer {
  //Kw カーネル数
  //IH　入力の高さ
  //IW　入力の幅
  //IC　入力チャネル数
  //OC　出力チャネル数
// val rand = new util.Random(0)
 // type T = Float
  val OH = (IH - KW) / st + 1
  val OW = (IW - KW) / st + 1
  var K = Array.ofDim[T](OC * IC * KW * KW)
  var dk = Array.ofDim[T](OC * IC * KW * KW)

  val KRMS = new RMSProp(OC * IC * KW * KW , 1 ,e,rhot1,rhot2, c)
  val KA = new Adam(OC * IC * KW * KW , 1, e,rhot1,rhot2, c)
  
  var Klist = List[Array[T]]()
  var Xlist1 = List[Array[T]]()
  var Xlist2 = List[Array[Array[T]]]()

  val encode = "UTF-8"
  val append = false

  var n = 0

  def push_K1D(V:Array[T]) = { Klist ::= V; V }
  def pop_K1D() = { val V = Klist.head; Klist = Klist.tail; V }

  def push_X1D(V:Array[T]) = { Xlist1 ::= V; V }
  def pop_X1D() = { val V = Xlist1.head; Xlist1 = Xlist1.tail; V }

  def push2D(V:Array[Array[T]]) = { Xlist2 ::= V; V }
  def pop2D() = { val V = Xlist2.head; Xlist2 = Xlist2.tail; V }


  // 必要なパラメータを定義する ★
  def is(i:Int, j:Int, k:Int) = i*IH*IW+j*IW+k
  def os(i:Int, j:Int, k:Int) = i*OH*OW + j*OW + k
  def ks(i:Int, j:Int, k:Int, l:Int) = i * IC * KW * KW + j * KW * KW + k * KW + l
  def prof() = {
    "Convolution(" + KW +" " + IH +" " + IW + " " + IC + " " + OC + ")"
  }
  
  

  def save(pass : String) = {
    mkdir(pass)
    val kdata = pass + "/k.txt"

    val kf = new FileStream(kdata, append)
    val kp = new StreamWriter(kf, encode)

    for(i <- 0 until K.size){
      kp.write(K(i) + ",")
    }
     kp.close()
  }

  def load_h(h:DenseVector[T]) = {
   
  }

  def loadparameter(pass:String) = {
    var fk = scala.io.Source.fromFile(pass + "k.txt").getLines.toArray
    val k = fk(0).split(",").toArray

    for(i <- 0 until K.size){
      K(i) = k(i).toFloat
    }
  }

  override def load(pass:String) = {
    var fk = scala.io.Source.fromFile(pass + "k.txt").getLines.toArray
    val k = fk(0).split(",").toArray

    for(i <- 0 until k.size){
      K(i) = k(i).toFloat
    }
   // K = T(K , OC , IC * KW * KW)
  }

  def init() = {
    K = K.map(_ => rand.nextGaussian * af).map(a => a.toFloat)
  }

   def forward(x:DenseVector[T]) = {
    x
  }

  def backward(d:DenseVector[T]) = {
    d
  }

  def T(x:Array[T] , H:Int , W:Int) = {
    var y = Array.ofDim[T](H * W)
    var count = 0
    for(i <- 0 until W ; j <- 0 until H){
      y(count) = x(j * W + i)
      count += 1
    }
    y
  }
/*
  def Transpose(x:Array[T]) = {
    val y = Array.ofDim[T](OC * IC * KW * KW)
    for(i <- 0 until IC ; j <- 0 until KW; k <- 0 until KW ; l <- 0 until OC){
      y(i * KW * KW * OC + j * KW * OC + k * OC + l) = x(l * KW * KW * IC + i * KW * KW + j * KW + k)
    }
    y
  }
 */

   def Transpose(x:Array[T]) = {
    val y = Array.ofDim[T](OC * IC * KW * KW)
    for(i <- 0 until IC ; j <- 0 until KW; k <- 0 until KW ; l <- 0 until OC){
      y(i * KW * KW * OC + j * KW * OC + k * OC + l) = x(l * KW * KW * IC + i * KW * KW + j * KW + k)
    }
    y
  }

  def FTrans2D(x:Array[T] , dn:Int , OC:Int , OW:Int,OH:Int) = {
    var xs = Array.ofDim[T](dn,OC * OW * OH)
    for(d <- 0 until dn;h <- 0 until OH;w <- 0 until OW;c <- 0 until OC){
      xs(d)(h * OW + c * OH * OW + w) = x(d * OH * OW * OC + h * OW * OC + w * OC + c)
    }
    xs
  }

  def Trans2D(x:Array[T] , H:Int , W:Int) = {
    var xs = Array.ofDim[T](H,W)
  //  x.map(a => println(a))
    for(i <- 0 until H ; j <- 0 until W){
      xs(i)(j) =  x(i * W + j)
    }
    xs
  }

  def BTrans2D(x:Array[T] , H:Int , W:Int , K:Int) = {
    var xs = Array.ofDim[T](H,W * K)
    for(i <- 0 until H ; j <- 0 until W ; k <- 0 until K){
      xs(i)(j * K + k) = x(i *W * K + j * K + k)
    }
    xs
  }

  def convert(x:Array[Array[T]]) = {
    var y = Array.ofDim[T](x.size * x(0).size)    
    for(i <- 0 until x.size ;j <- 0 until OH * OW; k <- 0 until OC){
      y(i * OH * OW * OC + j * OC + k) = x(i)(k * OH * OW + j)
    }
    y
  }

  def lowering(x:Array[Array[T]]) = {
    val y = Array.ofDim[T](x.size * KW * KW * OW * OH * IC)
    var count = 0

    for(b <- 0 until x.size; j <- 0 until OH; k <- 0 until OW) {
      for(l <- 0 until IC; m <- 0 until KW; n <- 0 until KW) {
        y(count) = x(b)(is(l,j+m,k+n))
        count += 1
      }
    }
    y
  }

  def relowering(x:Array[T] , dn:Int) = {
    var y = Array.ofDim[T](dn , IC * IH * IW)
    var count = 0
    for(n <- 0 until dn){
      for(i <- 0 until OH ; j <- 0 until OW ; c <- 0 until IC){
        for(p <- 0 until KW ; q <- 0 until KW){
        y(n)(c * IH * IW + (p + i) * IW + (q + j)) += x(count)
          count += 1
        }
      }
    }
    y
  }


  def forward(x:Array[T]) = {
    var y = Array.ofDim[T](OC * OW * OH)
   
    val lx = lowering(Array(x)) //行列にする
    val tk = T(K , OC , IC * KW * KW)
   
    push_X1D(lx)
    push_K1D(tk)
    libsgemm.BLAS.matmul(lx ,tk ,y ,OW * OH ,OC,KW * KW * IC)
    val y1 = FTrans2D(y,1,OC,OW , OH)
    y1.flatten
  }

   def backward(G:Array[T]) = {
    val V1 = pop_X1D()
    n += 1
    for(i <- 0 until OC){
      for(j <- 0 until IC){
        for(k <- 0 until KW){
          for(l <- 0 until KW){
            for(m <- 0 until OH if((m % st) == 0)){
              for(n <- 0 until OW if((n % st) == 0)){
                dk(ks(i,j,k,l))  += G(os(i,m,n)) * V1(is(j,k+m,l+n))
              }
            }
            
          }
        }
      }
    }

    var dV = Array.ofDim[T](IC * IH * IW)
    for(i <- 0 until IC; j <- 0 until IH; k <- 0 until IW) {
      for(l <- 0 until OH if((l % st) == 0); m <- 0 until KW if l + m == j) {
        for(n <- 0 until OW if((n % st) == 0); p <- 0 until KW if n + p == k) {
          for(q <- 0 until OC) {
            dV(is(i,j,k)) += K(ks(q,i,m,p)) * G(os(q,l,n))
          }
        }
      }
    }
    dV
  }
 

  def forward(x:Array[Array[T]]) = {
    var y = Array.ofDim[T](x.size * OC * OW * OH)
   
    val lx = lowering(x) //行列にする
   
    val tk = T(K , OC , IC * KW * KW)
   
    push_X1D(lx)
    push_K1D(tk)
    libsgemm.BLAS.matmul(lx ,tk ,y ,OW * OH * x.size,OC,KW * KW * IC)
    val y1 = FTrans2D(y,x.size,OC,OW , OH)
    y1
  }

  def backward(G:Array[Array[T]]) = {
    val V1 = pop_X1D()
    val Kt = pop_K1D()
    val D = convert(G)
    n += G.size
    var y = Array.ofDim[T](G.size * OH * OW * IC * KW * KW)
    val tk = T(Kt , IC * KW * KW , OC)
    val xt = T(V1 , OH * OW * G.size , KW * KW * IC)
    libsgemm.BLAS.matmul(xt,D,dk,KW * KW * IC, OC , OH * OW * G.size)
    libsgemm.BLAS.matmul(D,tk,y ,OW * OH * G.size,IC * KW * KW,OC)
    val y1 = relowering(y , G.size)

    y1
  }
  
  def update_Adam() = {
    // Kを更新する ★
    K = KA.update(K , dk , 0)
    //println("convolution_Update")
    reset()
  }

  def update_RMSProp() = {
    K = KRMS.update(K , dk , 0)
    reset()
  }

  def update() = {
  }

  /*
   def update() {
   // kを更新する ★
   val lr = 0.001
   for(i <- 0 until dk.size){
   K(i) -= lr * dk(i)
   }
   reset()
   }
   */
  def reset() {
    // dKを初期化する ★
    dk = Array.ofDim[T](OC * IC * KW * KW)
     Klist = List[Array[T]]()
     Xlist1 = List[Array[T]]()
     Xlist2 = List[Array[Array[T]]]()

    n=0
     }
}
 */
