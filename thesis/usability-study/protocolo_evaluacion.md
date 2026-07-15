# Protocolo de evaluación de usabilidad y aplicabilidad — kubescan

Estudio de evaluación exigido por el TFE de tipo *Desarrollo de software*
(instrucciones §2.6, Capítulo «Evaluación»: evaluación de usabilidad y
aplicabilidad con usuarios expertos).

---

## 1. Objetivo del estudio

Evaluar, con usuarios expertos, (a) la **usabilidad** de la herramienta
`kubescan` (facilidad de instalación, ejecución e interpretación de resultados)
y (b) su **aplicabilidad** al flujo de trabajo real de auditoría de seguridad de
clústeres *Kubernetes* / integración en *pipelines* CI/CD.

## 2. Participantes

- **Tamaño de muestra: 5–6 usuarios expertos** (mínimo defendible 5; 3 solo si
  la captación resulta imposible, documentándolo como amenaza a la validez).
  Justificación: Nielsen (2000) — ~5 usuarios detectan ~85 % de los problemas de
  usabilidad en pruebas formativas; con menos, el sesgo por participante domina
  la media SUS.
- **Perfil de «experto»** (debe cumplir al menos uno):
  - Ingeniero/a DevOps, SRE o *platform engineer* con experiencia operando
    *Kubernetes*.
  - Profesional de seguridad (AppSec, *cloud security*, *pentesting*) con
    exposición a contenedores/*Kubernetes*.
  - Requisito mínimo: ≥1 año trabajando con *Kubernetes* o seguridad de
    contenedores, y familiaridad con la línea de comandos.
- Registrar el perfil de cada participante (rol, años de experiencia con K8s,
  familiaridad con CI/CD y con seguridad de contenedores) — ver cuestionario.

## 3. Materiales y entorno

- Un equipo (o VM) con **Python 3.10+** y acceso a internet.
- El repositorio del proyecto disponible localmente.
- Tres directorios de manifiestos de ejemplo, provistos en `muestras/`:
  - `01_limpio/` — clúster bien configurado (veredicto esperado: CLEAN).
  - `02_misconfig/` — misconfiguraciones aisladas (veredicto esperado:
    ISOLATED_MISCONFIG).
  - `03_cadena/` — pod privilegiado + cuenta de servicio (veredicto esperado:
    ATTACK_CHAIN).
- Cronómetro (opcional, para tiempos por tarea) y hoja de observación.
- El estudio puede realizarse presencial o remoto (compartición de pantalla).

## 4. Procedimiento (≈25–30 min por participante)

1. **Bienvenida y consentimiento** (2 min). Explicar el objetivo, que se evalúa
   la herramienta y no a la persona, y recoger el consentimiento (cuestionario,
   sección 0). Pedir que piensen en voz alta (*think-aloud*).
2. **Datos de perfil** (2 min). Cuestionario, sección 1.
3. **Tareas** (15–20 min). El facilitador NO ayuda salvo bloqueo total; anota
   éxito, incidencias y (opcional) tiempo. Ver §5.
4. **Cuestionario post-test** (5 min). SUS (sección 2) + aplicabilidad
   (sección 3) + preguntas abiertas (sección 4).
5. **Cierre** (1 min). Agradecer y recoger comentarios finales.

## 5. Tareas

| # | Tarea | Éxito = | RNF/objetivo relacionado |
|---|-------|---------|--------------------------|
| T1 | **Instalar** kubescan (`pip install -e kubescan/`) y comprobar que el comando `kubescan` responde | comando disponible | RNF-3 Instalabilidad |
| T2 | **Escanear** `muestras/03_cadena/` e indicar el **veredicto** y la puntuación *ensemble* | identifica ATTACK_CHAIN y la puntuación | Uso básico |
| T3 | **Interpretar**: en `muestras/02_misconfig/`, con `--show-nodes`, indicar el manifiesto de mayor riesgo y **qué flag de escape** lo explica | identifica el manifiesto y la flag | Interpretabilidad del informe |
| T4 | **Salida JSON**: ejecutar con `--format json` sobre `muestras/03_cadena/` y extraer `ensemble_score` y `n_escape_capable` | extrae ambos campos | Integración CI/CD (RNF-4) |
| T5 | *(opcional, aplicabilidad)* escanear un directorio de manifiestos **propio/real** y valorar el resultado | ejecuta y comenta | Validez externa |

Métricas por tarea (hoja de recogida): **éxito** (logrado / logrado-con-ayuda /
no logrado), **tiempo** (opcional) e **incidencias** observadas.

## 6. Métricas e instrumentos

- **Éxito por tarea** → tasa de éxito (%) por tarea y global.
- **SUS (System Usability Scale)**, 10 ítems, escala 1–5 → puntuación 0–100.
  - *Scoring*: ítems impares (1,3,5,7,9) puntúan `valor − 1`; ítems pares
    (2,4,6,8,10) puntúan `5 − valor`; sumar los 10 (0–40) y multiplicar por 2,5.
  - *Interpretación* (Bangor et al., 2009; Sauro): media de referencia **68**;
    ≥80,3 «excelente» (A), 68–80,3 «bueno», ~68 «aceptable», <51 «pobre».
- **Aplicabilidad**: 4 ítems Likert 1–5 + preguntas abiertas (sección 3–4).
- **Cualitativo**: notas *think-aloud*, agrupadas en temas (lo más útil, mayores
  fricciones, mejoras propuestas).

## 7. Análisis y volcado en la memoria (Capítulo «Evaluación»)

Rellenar `plantilla_resultados.csv` y reportar en el TFE:

1. **Tabla — perfil de participantes** (rol, años K8s, CI/CD, seguridad).
2. **Tabla — tasa de éxito por tarea** (y global).
3. **SUS**: media ± DE, puntuación por participante, interpretación frente al
   umbral 68 y a la escala de adjetivos. (Figura de barras opcional.)
4. **Aplicabilidad**: medias Likert por ítem + síntesis cualitativa.
5. **Amenazas a la validez**: N reducido, sesgo de selección, efecto facilitador.

> Los resultados deben ser **datos reales recogidos**; este protocolo y el
> cuestionario pueden incluirse como **Anexo** (instrucciones §1.2 permiten
> cuestionarios/encuestas en anexos).

---

### Referencias del método
- Nielsen, J. (2000). *Why You Only Need to Test with 5 Users*.
- Brooke, J. (1996). *SUS: A quick and dirty usability scale*.
- Bangor, A., Kortum, P., & Miller, J. (2009). *Determining what individual SUS
  scores mean: Adding an adjective rating scale*.
