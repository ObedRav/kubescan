# Formulario de evaluación — cómo crearlo

## Opción rápida (recomendada): Apps Script — lo genera automáticamente
Usa `crear_formulario.gs`: abre https://script.google.com, pega el archivo,
ejecuta `crearFormulario` una vez y autoriza. Crea el formulario completo
(perfil, tareas, SUS, aplicabilidad, abiertas) y registra la URL para responder.
No requiere CLI ni credenciales locales. Los títulos ya vienen prefijados
(`SUS1`, `A1`, `T1`…) para que la exportación a CSV encaje con
`compute_results.py`.

> ¿Hay un CLI? La Google Forms API permite crearlo por código, pero exige un
> proyecto de Google Cloud + OAuth; para un formulario único, Apps Script es más
> rápido de poner en marcha.

## Opción manual: guion para construirlo a mano

Crear una sola vez (≈10 min) y reutilizar el enlace con los 3 participantes.
Todas las preguntas **obligatorias**. Al final, «Respuestas → Exportar a CSV»,
renombrar las columnas al formato de `plantilla_resultados.csv` y ejecutar
`python3 compute_results.py plantilla_resultados.csv`.

---

**Título del formulario:** Evaluación de usabilidad — kubescan
**Descripción:** Evaluamos la herramienta, no a la persona. ~4 minutos. Respuestas anónimas, uso académico (TFE).

## Sección 1 — Perfil
1. Rol *(opción única)*: DevOps/SRE · Platform engineer · Seguridad/AppSec · Desarrollo/IA · Otro
2. Años con Kubernetes *(opción única)*: <1 · 1–3 · 3–5 · >5
3. Familiaridad con seguridad de contenedores *(única)*: Baja · Media · Alta
4. Uso de CI/CD *(única)*: Nunca · Ocasional · Habitual

## Sección 2 — Resultado de las tareas *(opción única por tarea: Logrado / Logrado con ayuda / No logrado)*
- T1 Instalación · T2 Escaneo básico · T3 Interpretación · T4 Salida JSON · T5 Caso propio (o «No aplica»)

## Sección 3 — SUS *(escala lineal 1–5; 1 = totalmente en desacuerdo, 5 = totalmente de acuerdo)*
Usar exactamente los 10 enunciados de `GUIA_PARTICIPANTE.md` (SUS 1–10), en ese orden.

## Sección 4 — Aplicabilidad *(escala lineal 1–5)*
A1, A2, A3, A4 (enunciados en `GUIA_PARTICIPANTE.md`).

## Sección 5 — Abiertas *(respuesta larga)*
- ¿Lo más útil?  · ¿Qué cambiarías / mayor fricción?  · ¿Un caso de uso real?  · Comentarios.

---

### Correspondencia de columnas para `plantilla_resultados.csv`
`participant, role, years_k8s, cicd, sec_familiarity, T1_success..T5_success (1 / 0.5 / 0),
SUS1..SUS10 (1–5), A1..A4 (1–5)`.
El *scoring* SUS (impares valor−1, pares 5−valor, ×2,5) lo calcula
`compute_results.py`; no hay que hacerlo a mano.
