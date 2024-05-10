using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.Dnn;

namespace ObjectsDetection;

    class ObjectDetection
    {

        public void RunDetection()
        {

        // Yolo Object Detection (selecionando a CNN)
        // DNN -> Deep Neural Network, "invocando" o modelo
        var net = Emgu.CV.Dnn.DnnInvoke.ReadNetFromDarknet("./detection/yolov3.cfg", "./detection/yolov3.weights");

        // Labeling
        var classLabels = File.ReadAllLines("./detection/coco.names");

        // Usar lógica do OpenCv e processar por meio da CPU
        net.SetPreferableBackend(Emgu.CV.Dnn.Backend.OpenCV);
        net.SetPreferableTarget(Emgu.CV.Dnn.Target.Cpu);

        var vc = new VideoCapture(0, VideoCapture.API.DShow);

        // Matriz de imagem do vídeo
        Mat frame = new();

        // Matriz de resultado da CNN
        VectorOfMat output = new();

        // Boxes
        VectorOfRect boxes = new();

        // Acurácia
        VectorOfFloat scores = new();

        VectorOfInt indices = new();

        while (true)
        {
            // Fazer a captura de vídeo "ler" o frame
            vc.Read(frame);

            // Tornar o processamento mais rápido  
            CvInvoke.Resize(frame, frame, new System.Drawing.Size(0, 0), .4, .4);
            // Instanciar novamente os objetos
            boxes = new();
            indices = new();
            scores = new();
         
            // Convertendo a cadeia(bidimensional) de bytes para BGR
            var image = frame.ToImage<Bgr, byte>();

            // 1/255.0 para criar um blob pequeno (Drawing Size)
            // No OpenCv, os canais red e blue são trocados(por isso, BGR), então é necessário trocar novamente para RGB
            var input = DnnInvoke.BlobFromImage(image, 1/255.0, swapRB: true);

            // Input para a CNN
            net.SetInput(input);

            //Tagging -> Marcar, colocar um retângulo ao redor
            //Labeling -> Descrever ("gato", "celular", ...)

            // Obter resultado sem labeling
            net.Forward(output, net.UnconnectedOutLayersNames);
           
            //Percorrer os resultados
            for (int i = 0; i < output.Size; i++)
            {
                
                // Selecionando o valor i da matriz de resultados
                var mat = output[i];

                // Cast para obter Array bidimensional de floats (mat.GetData() retorna os valores em um array)
                var data = (float[,])mat.GetData();

                // For para selecionar as linhas, que contêm a acurácia, a classe identificadora do objeto
                for (int j = 0; j < data.GetLength(0); j++)
                {
                    // Selecionando todas as informações da linha (j é o mesmo, x é incrementado)
                    float[] row = Enumerable.Range(0, data.GetLength(1))
                                  .Select(x => data[j, x])
                                  .ToArray();
                   
                    // Cada linha é um "registro" de dados

                    // Ignorando 5 pois possui outras informações
                    var rowScore = row.Skip(5).ToArray();

                    // rowScore possui várias pontuações para a acurácia das classes, selecionamos o id da classe de maior pontuação
                    var classId = rowScore.ToList().IndexOf(rowScore.Max());

                    // A acurácia, de fato
                    var confidence = rowScore[classId];

                    // Se estiver "certo" o suficiente, tagging
                    if (confidence > 0.8f)
                    {
                        
                        // Localização do objeto detectado

                        //Centro do eixo X
                        var centerX = (int)(row[0] * frame.Width);

                        // Centro do eixo Y
                        var centerY = (int)(row[1] * frame.Height);

                        // Dimensões da tag
                        var boxWidth = (int)(row[2] * frame.Width);
                        var boxHeight = (int)(row[3] * frame.Height);

                        // Obter coordenadas de a partir de onde do objeto a tag será representada
                        var x = (int)(centerX - boxWidth / 2);
                        var y = (int)(centerY - (boxHeight / 2));

                        // Desenhar o retângulo nas coordenadas x e y, com largura w e altura h
                        boxes.Push(new System.Drawing.Rectangle[] { new System.Drawing.Rectangle(x, y, boxWidth, boxHeight) });

                        // Adicionar ao retângulo o id da classe e a pontuação de fidelidade
                        indices.Push(new int[] { classId });
                        scores.Push(new float[] { confidence });

                    }
                }
            }


            // Non-Max Suppression Algorithm (método de DNN)
            // Selecionamos todos as tags anteriormente e agora vamos ter um array com os melhores(é possível ter mais de uma tag na tela ao mesmo tempo)
            // NMSBoxces -> "Perfoms Non-Max Suppression given boxes and corresponding scores"
            // Limita para que o menor score da tag apresentado seja no máximo 0.8
            var bestIndex = DnnInvoke.NMSBoxes(boxes.ToArray(), scores.ToArray(), .8f, .8f);

            //Criar o frame de output
            var frameOut = frame.ToImage<Bgr, byte>();

            
            for (int i = 0; i < bestIndex.Length; i++)
            {
                // Posição do melhor índice(de maior pontuação)
                int index = bestIndex[i];

                // Selecionando qual das boxes representa o índice de maior pontuação
                var box = boxes[index];

                // Exibindo um retângulo verde na box de maior pontuação
                CvInvoke.Rectangle(frameOut, box, new MCvScalar(0, 255, 0), 2);

                // Processo de labeling, o nome da classe estará em vermelho e posicionado 10 pixels abaixo da altura da tag
                CvInvoke.PutText(frameOut, classLabels[indices[index]], new System.Drawing.Point(box.X + 5, box.Y - 5), Emgu.CV.CvEnum.FontFace.HersheyPlain, 1.0, new MCvScalar(0, 255, 0), 1);
                
            }

            // Todo o processamento foi feito com uma imagem reduzida, voltar à escala original
            CvInvoke.Resize(frameOut, frameOut, new System.Drawing.Size(0, 0), 4, 4);

            CvInvoke.Imshow("Object Detection", frameOut);

            if (CvInvoke.WaitKey(1) == 27)
                break;
            

        }
    }

}



    


