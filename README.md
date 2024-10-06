Evaluación de LLMs en conjunto con actividades lúdicas para usuarios que sufran de soledad y/o ansiedad. TFG UOC 2023-24 S2, Ingenieria Informática, rama de computación. 

Video para verlo en funcionamiento: https://vimeo.com/1016594006?share=copy#t=0

Chatngame ("Chat 'n' Game", en castellano "charla y juego") sería el programa lanzador.

Se necesita además:
- API key de Cohere, lograble tras registrarse en https://cohere.com/ . Editar "online_config.py", es el archivo que la usa. El programa usa "command", uno de los modelos de lenguaje gratuitos disponibles.
- modelo GPTQ cuantizado de Llama 2: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GPTQ
- modelo GPTQ cuantizado de Mistral 7B Instruct v0.2 https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GPTQ

Y una tarjeta gráfica Nvidia con CUDA, con al menos unos 8GB de VRAM.
