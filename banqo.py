from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

# Argumentos a serem informados por meio do terminal
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", type=str, default="mobilenet_ssd/MobileNetSSD_deploy.prototxt", help="Caminho para o arquivo do arquivo prototxt do framework Caffe")
ap.add_argument("-m", "--model", type=str, default="mobilenet_ssd/MobileNetSSD_deploy.caffemodel", help="Caminho para o modelo de CNN pré-treinado Caffe")
ap.add_argument("-i", "--input", type=str, help="Caminho para o vídeo de input [opcional]")
ap.add_argument("-o", "--output", type=str, help="Caminho para o vídeo de output [opcional]")
ap.add_argument("-c", "--confidence", type=float, default=0.6, help="Probabilidade mínima para filtragem de detecções fracas")
ap.add_argument("-s", "--skip-frames", type=int, default=30, help="Número de quadros a serem pulados entre detecções")
args = vars(ap.parse_args())

# Inicializa a lista de classificações que o algoritmo de rede de aprendizado profundo foi treinado para detectar
CLASSES = [
	"background",
	"aeroplane",
	"bicycle",
	"bird",
	"boat",
	"bottle",
	"bus",
	"car",
	"cat",
	"chair",
	"cow",
	"diningtable",
	"dog",
	"horse",
	"motorbike",
	"person",
	"pottedplant",
	"sheep",
	"sofa",
	"train",
	"tvmonitor"
]

# Carrega o modelo serializado informado pelos parâmetros
print("[INFO] Carregando modelo...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Caso um vídeo não tenha sido informado, utilizar as imagens da webcam
if not args.get("input", False):
	print("[INFO] Iniciando stream de vídeo...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)
else:
	print("[INFO] Abrindo arquivo de vídeo...")
	vs = cv2.VideoCapture(args["input"])

# Garante que nenhuma divisão por 0 será realizada
if args["skip_frames"] == 0:
	args["skip_frames"] = 1
	print(args["skip_frames"])

# Inicializa o gravador de vídeo - que será inicializado posteriormente se necessário
writer = None 

# Inicializa as dimensões do vídeo
W = None
H = None

# Inicializa uma instância do rastreador de centróides, considerando um objeto não detectado por mais de 20 quadros como desaparecido 
ct = CentroidTracker(maxDisappeared=25, maxDistance=50)
# Inicializa a lista de armazenamento de cada um dos rastreadores de correlação dlib
trackers = [] 
# Inicializa um dictionary para registrar cada ID de um objeto com uma instância da classe TrackableObject
trackableObjects = {} 

# Inicializa o total de quadros processados até o momento
totalFrames = 0 

# Inicia o estimador da taxa de transferência de quadros
fps = FPS().start() 

# Executa o loop enquanto houverem quadros da stream de vídeo
while True:
	# Instancia o próximo quadro a ser processado
	frame = vs.read() 
	# Define se o quadro será lido pelo VideoCapture ou VideoStream
	frame = frame[1] if args.get("input", False) else frame 

	# Caso nenhum quadro tenha sido instanciado, declara-se o fim do vídeo
	if args["input"] is not None and frame is None:
		break

	# Redimensiona o quadro para ter um máximo de 500 pixels de largura por uma maior velocidade de processamento
	frame = imutils.resize(frame, width=500) 
	# Converte o quadro de BGR para RGB para a biblioteca dlib
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

	# Instancia as dimensões do quadro caso necessário
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# Se foi informada a vontade do usuário de registrar o resultado, inicializa o gravador de vídeo com o caminho desejado
	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

	# Inicializa o status atual do processo
	status = "Esperando" 
	# Inicializa uma lista de armazenamento dos retângulos delimitadores retornados pelo detector de objetos ou os rastreadores de correlação
	rects = [] 

	# Define se, no quadro atual, deve-se executar o método de detecção computacionalmente mais custoso para auxiliar o rastreador
	if totalFrames % args["skip_frames"] == 0:
		# Atualiza o status do processo para a fase de detecção ativa de objetos
		status = "Detectando" 
		# Inicializa o novo conjunto de rastreadores de objetos
		trackers = [] 

		# Converte o quadro para um blob
		blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5) 
		# Passa o blob pela rede
		net.setInput(blob) 
		# Recebe as detecções obtidas
		detections = net.forward() 

		# Executa o loop para cada detecção obtida
		for i in np.arange(0, detections.shape[2]):
			# Obtêm o percentual de certeza da detecção
			confidence = detections[0, 0, i, 2] 

			# Testa se o threshold de confiança informado tolera o percentual de certeza obtido
			if confidence > args["confidence"]:
				# Extrai o index da classificação dentre a lista de detecções
				idx = int(detections[0, 0, i, 1]) 

				# Ignora a classificação se ela não seguir os objetos procurados
				if CLASSES[idx] not in ["chair", "sofa"]:
					continue

				# Calcula as coordenadas (x, y) da caixa delimitadora do objeto
				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				(startX, startY, endX, endY) = box.astype("int")

				# Monta um objeto retangular da biblioteca dlib para delimitação das coordenadas
				rect = dlib.rectangle(startX, startY, endX, endY) 
				# Inicia o rastreador de correçações do dlib
				tracker = dlib.correlation_tracker()
				tracker.start_track(rgb, rect)
				
				# Adiciona o rastreador à lista de rastreadores para ser utilizado durante os quadros "ignorados"
				trackers.append(tracker) 

	# Do contrário, no quadro atual, serão utilizados os rastreadores ao invés de detecções de objetos para  uma melhor taxa de transferência de quadros
	else:
		# Executo o loop para cada rastreador
		for tracker in trackers:
			# Atualiza o status do processo para a fase de rastreamento de objetos no quadro
			status = "Tracking" 

			# Atualiza o rastreador
			tracker.update(rgb) 
			# Obtêm a posição
			pos = tracker.get_position() 

			# Extrai o objeto que armazena as coordenados
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			# Informa tais posições na lista de retângulos delimitadores
			rects.append((startX, startY, endX, endY)) 

	# Usa do rastreador de centróides para associar o antigo objeto de centróides com  os objetos recém-computados
	objects = ct.update(rects)


	# Executa o loop para cada objeto rastreado
	for (objectID, centroid) in objects.items():
		# Tenta obter uma instância de TrackableObject para o objectID dessa iteração
		to = trackableObjects.get(objectID, None) 

		# Caso não haja nenhuma, ela é criada
		if to is None:
			to = TrackableObject(objectID, centroid)

		# Registra o objeto rastreável no dictionary
		trackableObjects[objectID] = to 

		# Desenha as informações do objeto no quadro a ser exibido
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	# Monta uma lista de tuplas de informações que serão exibidas no quadro
	info = [
		("Contados", len(trackableObjects)),
	]

	# Executa o loop para cada tupla de informação e as desenha no quadro
	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	# Confere se deve-se registrar o quadro na memória secundária
	if writer is not None:
		writer.write(frame)

	# Mostra o quadro a ser exibido
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# Caso a tecla 'Q' seja pressionada, interromper o loop 
	if key == ord("q"):
		break

	# Incrementa a contagem de quadros total
	totalFrames += 1 
	# Atualiza o contador de FPS
	fps.update() 

# Interrompe o estimador da taxa de transferência de quadros
fps.stop() 

# Exibe as informações
print("[INFO] Tempo decorrido: {:.2f}".format(fps.elapsed()))
print("[INFO] FPS aproximado: {:.2f}".format(fps.fps()))
print("[INFO] Objetos contados: {:.2f}".format(len(trackableObjects)))

# Verifica se é necessário liberar o ponteiro do gravador de vídeo
if writer is not None:
	writer.release()

# Caso um arquivo de vídeo não esteja sendo usado, interrempe a stream de vídeo da câmera
if not args.get("input", False):
	vs.stop()

# Caso contrário, libera o ponteiro do arquivo de vídeo
else:
	vs.release()

# Fecha qualquer janela aberta
cv2.destroyAllWindows()