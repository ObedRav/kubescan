# kubescan — guía rápida de evaluación (≈10 minutos)

Gracias por participar. Evaluamos **la herramienta, no a ti**. Piensa en voz alta
si puedes. Al final respondes un breve cuestionario. Hazlo a tu ritmo y de forma
remota; no hace falta agendar nada.

---

## Antes de empezar (elige UNA vía)

**Vía rápida (recomendada, ~5 min de tareas).** Te facilitamos un entorno ya
preparado (una VM / sesión con la herramienta instalada). Solo ejecutas los
comandos de las tareas 2–4. Salta el Paso 0.

**Vía autónoma (~15 min).** Instalas tú mismo (Paso 0) y luego las tareas.

### Paso 0 — Instalación (solo vía autónoma) · Tarea T1
Requisitos: Python 3.10+, git.
```bash
git clone https://github.com/ObedRav/kubescan.git
cd kubescan
python3 -m venv .venv && source .venv/bin/activate
pip install -e kubescan/
kubescan --help          # debe mostrar la ayuda (comandos scan / live)
```
*Éxito de T1:* el comando `kubescan` responde.

---

## Tareas (ejecuta y anota lo que observas)

> En la vía rápida, sitúate en la raíz del repositorio y ejecuta directamente.

### Tarea T2 — Escaneo básico
```bash
kubescan scan thesis/usability-study/muestras/03_cadena
```
Anota: **veredicto** y **puntuación (Ensemble score)**.

### Tarea T3 — Interpretación
```bash
kubescan scan thesis/usability-study/muestras/02_misconfig --show-nodes
```
Anota: de los **dos** manifiestos, ¿cuál es el de **mayor riesgo** y qué
**flags** explican su riesgo?

### Tarea T4 — Salida para automatización (JSON)
```bash
kubescan scan thesis/usability-study/muestras/03_cadena --format json
```
Anota: los valores de `ensemble_score` y `n_escape_capable`.

### Tarea T5 — (opcional) tu propio caso
Si tienes a mano un directorio de manifiestos propios, escanéalo y comenta el
resultado:
```bash
kubescan scan /ruta/a/tus/manifiestos
```

---

## Cuestionario (≈4 min)

Rellena el formulario que te hemos enviado: 

*(Si prefieres, responde aquí mismo con los números y nos lo devuelves.)*

**Perfil:** rol · años con Kubernetes · familiaridad con seguridad de
contenedores (baja/media/alta) · uso de CI/CD (nunca/ocasional/habitual).

**SUS — grado de acuerdo, 1 = totalmente en desacuerdo … 5 = totalmente de acuerdo:**
1. Usaría esta herramienta con frecuencia.
2. La herramienta es innecesariamente compleja.
3. La herramienta es fácil de usar.
4. Necesitaría apoyo técnico para poder usarla.
5. Las funciones están bien integradas.
6. La herramienta es demasiado inconsistente.
7. La mayoría aprendería a usarla con rapidez.
8. Es incómoda/engorrosa de usar.
9. Me siento seguro/a usándola.
10. Necesité aprender muchas cosas antes de empezar.

**Aplicabilidad (1–5):**
- A1. La integraría en mi flujo de CI/CD o auditoría.
- A2. La priorización de riesgo (ranking) se ajusta a cómo auditaría un clúster.
- A3. El informe es accionable (sé qué revisar y por qué).
- A4. Confiaría en el veredicto para decidir revisar un clúster.

**Abiertas:** lo más útil · lo que cambiarías · un caso de uso real.

¡Gracias! Tiempo total estimado: ~10 minutos.
