# USAGE
# To read and write back out to video:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4 \
#	--output output/output_01.avi
#
# To read from webcam and write back out to disk:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
#	--output output/webcam_output.avi


#yay -S python-dlib

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
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str, help="path to optional input video file")
ap.add_argument("-o", "--output", type=str, help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4, help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30, help="# of skip frames between detections")
args = vars(ap.parse_args())

# Inicializa a lista de classificações que o algoritmo de rede de aprendizado profundo foi treinado para detectar
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

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

writer = None # Inicializa o gravador de vídeo - que será inicializado posteriormente se necessário

# Inicializa as dimensões do vídeo
W = None
H = None

ct = CentroidTracker(maxDisappeared=40, maxDistance=50) # Inicializa uma instância do rastreador de centróides, considerando um objeto não detectado por mais de 40 quadros como desaparecido 
trackers = [] # Inicializa a lista de armazenamento de cada um dos rastreadores de correlação dlib
trackableObjects = {} # Inicializa um dictionary para registrar cada ID de um objeto com uma instância da classe TrackableObject

totalFrames = 0 # Inicializa o total de quadros processados até o momento

fps = FPS().start() # Inicia o estimador da taxa de transferência de quadros

# Executa o loop enquanto houverem quadros da stream de vídeo
while True:
	frame = vs.read() # Instancia o próximo quadro a ser processado
	frame = frame[1] if args.get("input", False) else frame # Define se o quadro será lido pelo VideoCapture ou VideoStream

	# Caso nenhum quadro tenha sido instanciado, declara-se o fim do vídeo
	if args["input"] is not None and frame is None:
		break

	frame = imutils.resize(frame, width=500) # Redimensiona o quadro para ter um máximo de 500 pixels de largura por uma maior velocidade de processamento
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Converte o quadro de BGR para RGB para a biblioteca dlib

	# Instancia as dimensões do quadro caso necessário
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# Se foi informada a vontade do usuário de registrar o resultado, inicializa o gravador de vídeo com o caminho desejado
	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(W, H), True)

	status = "Esperando" # Inicializa o status atual do processo
	rects = [] # Inicializa uma lista de armazenamento dos retângulos delimitadores retornados pelo detector de objetos ou os rastreadores de correlação

	# Define se, no quadro atual, deve-se executar o método de detecção computacionalmente mais custoso para auxiliar o rastreador
	if totalFrames % args["skip_frames"] == 0:
		status = "Detectando" # Atualiza o status do processo para a fase de detecção ativa de objetos
		trackers = [] # Inicializa o novo conjunto de rastreadores de objetos

		blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5) # Converte o quadro para um blob
		net.setInput(blob) # Passa o blob pela rede
		detections = net.forward() # Recebe as detecções obtidas

		# Executa o loop para cada detecção obtida
		for i in np.arange(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2] # Obtêm o percentual de certeza da detecção

			# Testa se o threshold de confiança informado tolera o percentual de certeza obtido
			if confidence > args["confidence"]:
				idx = int(detections[0, 0, i, 1]) # Extrai o index da classificação dentre a lista de detecções

				# Ignora a classificação se ela não seguir os objetos procurados
				#if CLASSES[idx] != "bottle" or CLASSES[idx] != "sofa":
				if CLASSES[idx] != "person":
					continue

				# Calcula as coordenadas (x, y) da caixa delimitadora do objeto
				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				(startX, startY, endX, endY) = box.astype("int")

				'''
				# construct a dlib rectangle object from the bounding
				# box coordinates and then start the dlib correlation
				# tracker
				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				tracker.start_track(rgb, rect)
				'''

				rect = dlib.rectangle(startX, startY, endX, endY) # Monta um objeto retangular da biblioteca dlib para delimitação das coordenadas
				# Inicia o rastreador de correçações do dlib
				tracker = dlib.correlation_tracker()
				tracker.start_track(rgb, rect)
				
				trackers.append(tracker) # Adiciona o rastreador à lista de rastreadores para ser utilizado durante os quadros "ignorados"

	# Do contrário, no quadro atual, serão utilizados os rastreadores ao invés de detecções de objetos para  uma melhor taxa de transferência de quadros
	else:
		# Executo o loop para cada rastreador
		for tracker in trackers:
			# set the status of our system to be 'tracking' rather
			# than 'waiting' or 'detecting'
			status = "Tracking" # Atualiza o status do processo para a fase de rastreamento de objetos no quadro

			tracker.update(rgb) # Atualiza o rastreador
			pos = tracker.get_position() # Obtêm a posição

			# Extrai o objeto que armazena as coordenados
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			rects.append((startX, startY, endX, endY)) # Informa tais posições na lista de retângulos delimitadores

	#'''
	## draw a horizontal line in the center of the frame -- once an
	## object crosses this line we will determine whether they were
	## moving 'up' or 'down'
	#cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
	#'''

	# Usa do rastreador de centróides para associar o antigo objeto de centróides com  os objetos recém-computados
	objects = ct.update(rects)


	# Executa o loop para cada objeto rastreado
	for (objectID, centroid) in objects.items():
		to = trackableObjects.get(objectID, None) # Tenta obter uma instância de TrackableObject para o objectID dessa iteração

		# Caso não haja nenhuma, ela é criada
		if to is None:
			to = TrackableObject(objectID, centroid)

		#'''
		## otherwise, there is a trackable object so we can utilize it
		## to determine direction
		#else:
		#	# the difference between the y-coordinate of the *current*
		#	# centroid and the mean of *previous* centroids will tell
		#	# us in which direction the object is moving (negative for
		#	# 'up' and positive for 'down')
		#	y = [c[1] for c in to.centroids]
		#	direction = centroid[1] - np.mean(y)
		#	to.centroids.append(centroid)
		#	# check to see if the object has been counted or not
		#	if not to.counted:
		#		# if the direction is negative (indicating the object
		#		# is moving up) AND the centroid is above the center
		#		# line, count the object
		#		if direction < 0 and centroid[1] < H // 2:
		#			totalUp += 1
		#			to.counted = True
		#		# if the direction is positive (indicating the object
		#		# is moving down) AND the centroid is below the
		#		# center line, count the object
		#		elif direction > 0 and centroid[1] > H // 2:
		#			totalDown += 1
		#			to.counted = True
		#'''

		trackableObjects[objectID] = to # Registra o objeto rastreável no dictionary

		# Desenha as informações do objeto no quadro a ser exibido
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	# Monta uma lista de tuplas de informações que serão exibidas no quadro
	info = [
		("Contados", len(trackableObjects))
		#'''
		#("Up", totalUp),
		#("Down", totalDown),
		#("Status", status),
		#'''
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

	totalFrames += 1 # Incrementa a contagem de quadros total
	fps.update() # Atualiza o contador de FPD

fps.stop() # Interrompe o estimador da taxa de transferência de quadros

# Exibe as informações
print("[INFO] Tempo decorrido: {:.2f}".format(fps.elapsed()))
print("[INFO] FPS aproximado: {:.2f}".format(fps.fps()))
print('[INFO] Objetos contados: {:.2f}'.format(str(len(trackableObjects))))

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