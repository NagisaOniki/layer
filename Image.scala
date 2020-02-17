package layer
import breeze.linalg._
import pll.utils.RichArray._
object Image {
  type T = Float
  def load_mnist(dir:String) = {
    def fd(line:String) = line.split(",").map(_.toFloat / 256f).toArray
    def ft(line:String) = line.split(",").map(_.toInt).toArray
    val train_d = scala.io.Source.fromFile(dir + "/train-d.txt").getLines.map(fd).toArray
    val train_t = scala.io.Source.fromFile(dir + "/train-t.txt").getLines.map(ft).toArray.head
    val test_d = scala.io.Source.fromFile(dir + "/test-d.txt").getLines.map(fd).toArray
    val test_t = scala.io.Source.fromFile(dir + "/test-t.txt").getLines.map(ft).toArray.head
    (train_d.zip(train_t), test_d.zip(test_t))
  }
  def load_cifar(dir:String) = {
    def fd(line:String) = line.split(",").map(_.toFloat / 256f).toArray
    def ft(line:String) = line.split(",").map(_.toInt).toArray
    val train_d = scala.io.Source.fromFile(dir + "/train-d.txt").getLines.map(fd).toArray
    val train_t = scala.io.Source.fromFile(dir + "/train-t.txt").getLines.map(ft).toArray.head
    val test_d = scala.io.Source.fromFile(dir + "/test-d.txt").getLines.map(fd).toArray
    val test_t = scala.io.Source.fromFile(dir + "/test-t.txt").getLines.map(ft).toArray.head
    (train_d.map(encode(_)).zip(train_t), test_d.map(encode(_)).zip(test_t))
    //(train_d.zip(train_t), test_d.zip(test_t))
  }
  def encode(V:Array[T])={
    var V_new = new Array[T](V.size)
    var R = new Array[T](V.size/3)
    var G = new Array[T](V.size/3)
    var B = new Array[T](V.size/3)
    for(i<-0 until V.size){
      if(i%3 == 0){
        R(i/3) = V(i)
      }else if(i%3 == 1){
        G(i/3) = V(i)
      }else{
        B(i/3) = V(i)
      }
    }
    V_new = R ++ G ++ B
    V_new.toArray
  }
  def encode3D(V:Array[Array[Array[T]]])={
    //H:V.size---W:V(0).size---rgb:V(0)(0).size
    //-----RGBの順に並び替える処理----
    var R = List[T]()
    var G = List[T]()
    var B = List[T]()
    for(i<-0 until V.size){
      for(j<-0 until V(0).size){
        R ::= V(i)(j)(0)
        G ::= V(i)(j)(1)
        B ::= V(i)(j)(2)
      }
    }
      (R.reverse ++ G.reverse ++ B.reverse).toArray
  }

  def rgbSum(V:Array[Array[Array[T]]])={
    val H = V.size
    val W = V(0).size
    val C = V(0)(0).size //3
    var z = new Array[T](H*W)
    for(i<-0 until H ; j<-0 until W ; k<-0 until C){
      z(i*W+j) += V(i)(j)(k)
    }
    z
  }

  //卒論の学習データの背景の黒を白に変換
  def blackToWhite(V:Array[Array[Array[Int]]])={
    val H = V.size
    val W = V(0).size
    val C = V(0)(0).size
    var z = Array.ofDim[Int](H,W,C)
    for(i<-0 until H ; j<-0 until W){
      if(V(i)(j)(0) < 5 && V(i)(j)(1) < 5 && V(i)(j)(2) < 5){
        z(i)(j)(0) = 255
        z(i)(j)(1) = 255
        z(i)(j)(2) = 255
      }else{
        z(i)(j)(0) = V(i)(j)(0)
        z(i)(j)(1) = V(i)(j)(1)
        z(i)(j)(2) = V(i)(j)(2)
      }
    }
    z
  }


  //カラー画像の色の反転処理(encodeの後に行う)
  def invertedColor(xs:Array[Float])={
    xs.map(a=>256f-a)
  }

  //num:Duplication_num
  //read->dataOneToDuplication
  def dataOneToDuplication(x:Array[Array[Array[Int]]],num:Int)={
    var newdata = List[Array[Array[Array[Int]]]]()
    val H = x.size
    val W = x(0).size
    val C = x(0)(0).size
    val late = 10
    //rgb_Down
    var downlate = late
    for(n<-0 until num){
      var data = Array.ofDim[Int](H,W,C)
      for(i<-0 until H ; j<-0 until W ; k<-0 until C){
        if(x(i)(j)(k)-downlate >= 0){
          data(i)(j)(k) = data(i)(j)(k) - downlate
        }else{
          data(i)(j)(k) = data(i)(j)(k)
        }
      }
      newdata ::= data
      downlate += late
    }
    //rgb_Up
    var uplate = late
    for(n<-0 until num){
      var data = Array.ofDim[Int](H,W,C)
      for(i<-0 until H ; j<-0 until W ; k<-0 until C){
        if(x(i)(j)(k)-downlate >= 0){
          data(i)(j)(k) = data(i)(j)(k) + downlate
        }else{
          data(i)(j)(k) = data(i)(j)(k)
        }
      }
      newdata ::= data
      uplate += late
    }
    newdata.reverse.toArray
  }

  //リサイズを揃える関数
  def fill(V0:Array[Array[Array[T]]],H:Int,W:Int)={
    val V = Array.ofDim[T](H,W,3)
    for(i<-0 until V0.size){
      for(j<-0 until V0(0).size){
        for(k<-0 until V0(0)(0).size){
          V(i)(j)(k) = V0(i)(j)(k)
        }
      }
    }
    V
  }
  //画像の明暗を平均化
  def LightChange(V0:Array[Array[Array[T]]])={
    var R_ = 0f
    var G_ = 0f
    var B_ = 0f
    var V = Array.ofDim[T](V0.size,V0(0).size,V0(0)(0).size)
    //RGBそれぞれの平均計算
    for(i<-0 until V0.size){
      for(j<-0 until V0(0).size){
        R_ += V0(i)(j)(0) / (V0.size*V0(0).size)
        G_ += V0(i)(j)(1) / (V0.size*V0(0).size)
        B_ += V0(i)(j)(2) / (V0.size*V0(0).size)
      }
    }
    //RGB平均化
    for(i<-0 until V0.size){
      for(j<-0 until V0(0).size){
        V(i)(j)(0) = V0(i)(j)(0) - R_
        V(i)(j)(1) = V0(i)(j)(1) - G_
        V(i)(j)(2) = V0(i)(j)(2) - B_
      }
    }
    (V,R_,G_,B_)
  }
  //画像の明暗平均化を戻す
  def reLightChange(V0:Array[Array[Array[T]]],R:T,G:T,B:T)={
    var V = Array.ofDim[T](V0.size,V0(0).size,V0(0)(0).size)
    for(i<-0 until V0.size){
      for(j<-0 until V0(0).size){
        V(i)(j)(0) = V0(i)(j)(0) + R
        V(i)(j)(1) = V0(i)(j)(1) + G
        V(i)(j)(2) = V0(i)(j)(2) + B
      }
    }
    V
  }
  def rgb(im : java.awt.image.BufferedImage, i:Int, j:Int) = {
    val c = im.getRGB(i,j)
    Array(c >> 16 & 0xff, c >> 8 & 0xff, c & 0xff)
  }
  def pixel(r:Int, g:Int, b:Int) = {
    val a = 0xff
    ((a & 0xff) << 24) | ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff)
  }
  def read(fn:String) = {
    val im = javax.imageio.ImageIO.read(new java.io.File(fn))
    (for(i <- 0 until im.getHeight; j <- 0 until im.getWidth)
    yield rgb(im, j, i)).toArray.grouped(im.getWidth).toArray
  }
  def write(fn:String, b:Array[Array[Array[Int]]]) = {
    val w = b(0).size
    val h = b.size
    val im = new java.awt.image.BufferedImage(w, h, java.awt.image.BufferedImage.TYPE_INT_RGB);
    for(i <- 0 until im.getHeight; j <- 0 until im.getWidth) {
      im.setRGB(j,i,pixel(b(i)(j)(0), b(i)(j)(1), b(i)(j)(2)));
    }
    javax.imageio.ImageIO.write(im, "png", new java.io.File(fn))
  }
  def make_image(xs:Array[Array[T]] ,H:Int ,W:Int)={
    val im = Array.ofDim[Int]( H * 28 , W * 28, 3 ) //hight,width,RGB
    for(i<-0 until H; j<-0 until W){
      for(p<-0 until 28; q<-0 until 28; r<-0 until 3){
        im(i*28+p)(j*28+q)(r) = (xs(i*W+j)(p*28 + q)*255).toInt
      }
    }
    im
  }
  def make_colorimage(xs:Array[Array[T]],H:Int,W:Int,h:Int,w:Int)={
    val im = Array.ofDim[Int]( H*h , W*w , 3)
    for(i<-0 until H ; j<-0 until W){
      for(p<-0 until h ; q<-0 until w ; r<-0 until 3){
        im(i*w+p)(j*h+q)(r) = (xs(i*W+j)(p*w+q + r*h*w)*255).toInt
      }
    }
    im
  }

  def make_colorimageOne(xs:Array[Array[T]],h:Int,w:Int)={
    val im = Array.ofDim[T](h,w,3)
    for(p<-0 until h ; q<-0 until w ; r<-0 until 3){
      im(p)(q)(r) = (xs(0)(p*w+q + r*h*w)*255).toInt
    }
    im
  }



  //weather ver. (1次元ver.)
  def make_Wimage(xs:Array[T] ,H:Int ,W:Int,h:Int,w:Int)={
    val im = Array.ofDim[Int]( H*h , W*w , 3 )
    for(i<-0 until H; j<-0 until W){
      for(p<-0 until h; q<-0 until w; r<-0 until 3){
        im(i*w+p)(j*h+q)(r) = (xs(i*W+j + p*w + q)*255).toInt
      }
    }
    im
  }
  //上下または左右をきりとり正方形に整える
  def toSquare(xs:Array[Array[Array[T]]],H:Int,W:Int,h:Int,w:Int)={ //h,w:捨てる行数または列数
    var newxs = Array.ofDim[T](H,W,3)
    for(i<-0 until H ; j<-0 until W ; k<-0 until 3){
      newxs(i)(j)(k) = xs(i+3)(j)(k)
    }
    newxs
  }
  //----一次元を三次元（Int）に変換（引数はRGB順に詰められたもの）---------
  def to3DArrayOfColor(image:Array[T],h:Int,w:Int) = {
    val input = image.map(_*255)
    var output = List[Array[Array[T]]]()
    for(i <- 0 until h) {
      var row = List[Array[T]]()
      for(j <- 0 until w) {
        val red = input(i*w+j)
        val green = input(i*w+j+h*w)
        val blue = input(i*w+j+h*w*2)
        row ::= Array(red,green,blue)
      }
      output ::= row.reverse.toArray
    }
    output.reverse.toArray.map(_.map(_.map(_.toInt)))
  }
}
