Evaluación de LLMs en conjunto con actividades lúdicas para usuarios con ansiedad. TFG UOC 2023-24 S2, Ingenieria Informática, rama de computación.Chatngame ("Chat 'n' Game") sería el programa lanzador.

Se necesita además:
- API key, lograble tras registrarse en https://cohere.com/ .
Crear un archivo llamado online_config.py , que simplemente tenga esta linea con key_cohere siendo la key --> cohere_key = 'key_cohere'
- modelo GPTQ cuantizado de Llama 2: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GPTQ
- modelo GPTQ cuantizado de Mistral 7B Instruct v0.2 https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GPTQ

Y una tarjeta gráfica Nvidia con CUDA, con al menos unos 8GB de VRAM.
