/**
 * crear_formulario.gs — genera el formulario de evaluación de usabilidad de
 * kubescan en Google Forms, de una sola ejecucion.
 *
 * COMO USARLO (≈2 min, sin CLI ni credenciales):
 *   1. Abre https://script.google.com  ->  "Proyecto nuevo".
 *   2. Borra el contenido y pega TODO este archivo.
 *   3. En el desplegable de funciones elige "crearFormulario" y pulsa
 *      "Ejecutar" (Run). Autoriza el acceso a tu cuenta la primera vez.
 *   4. En "Registro de ejecucion" (Ver -> Registro) apareceran dos URLs: la de
 *      responder (para enviar a los 3 expertos) y la de editar.
 *
 * IMPORTANTE: este script SOLO se EJECUTA; NO uses "Implementar / Desplegar"
 * ni "Aplicacion web". No es una web app y no tiene doGet/doPost — desplegarlo
 * da el error "Script function not found: doGet". Solo Ejecutar.
 *
 * Las respuestas se ven en la pestana "Respuestas" del formulario; desde alli
 * "Vincular a Hojas de calculo" -> exportar a CSV -> renombrar columnas a
 * SUS1..SUS10 / A1..A4 / T1_success.. y ejecutar compute_results.py.
 */
function crearFormulario() {
  var form = FormApp.create('Evaluación de usabilidad — kubescan')
    .setDescription(
      'Evaluamos la herramienta, no a la persona. Duración ~10 minutos. ' +
      'Respuestas anónimas, uso exclusivamente académico (Trabajo Fin de Estudios). ' +
      'Realiza primero las tareas de la guía y luego responde este cuestionario.')
    .setProgressBar(true)
    .setCollectEmail(false)
    .setLimitOneResponsePerUser(false);

  var acuerdo = ['1 = Totalmente en desacuerdo', '5 = Totalmente de acuerdo'];
  var exito = ['Logrado', 'Logrado con ayuda', 'No logrado'];

  // --- Sección 1: Perfil ---
  form.addPageBreakItem().setTitle('Perfil profesional');
  form.addMultipleChoiceItem().setTitle('Rol').setRequired(true)
    .setChoiceValues(['DevOps/SRE', 'Platform engineer', 'Seguridad/AppSec', 'Desarrollo/IA', 'Otro']);
  form.addMultipleChoiceItem().setTitle('Años de experiencia con Kubernetes').setRequired(true)
    .setChoiceValues(['<1', '1–3', '3–5', '>5']);
  form.addMultipleChoiceItem().setTitle('Familiaridad con seguridad de contenedores').setRequired(true)
    .setChoiceValues(['Baja', 'Media', 'Alta']);
  form.addMultipleChoiceItem().setTitle('Uso de CI/CD en tu trabajo').setRequired(true)
    .setChoiceValues(['Nunca', 'Ocasional', 'Habitual']);

  // --- Sección 2: Resultado de las tareas ---
  form.addPageBreakItem().setTitle('Resultado de las tareas');
  form.addMultipleChoiceItem().setTitle('T1 — Instalación').setRequired(true).setChoiceValues(exito);
  form.addMultipleChoiceItem().setTitle('T2 — Escaneo básico').setRequired(true).setChoiceValues(exito);
  form.addMultipleChoiceItem().setTitle('T3 — Interpretación').setRequired(true).setChoiceValues(exito);
  form.addMultipleChoiceItem().setTitle('T4 — Salida JSON').setRequired(true).setChoiceValues(exito);
  form.addMultipleChoiceItem().setTitle('T5 — Caso propio (opcional)').setRequired(false)
    .setChoiceValues(['Logrado', 'Logrado con ayuda', 'No logrado', 'No aplica']);

  // --- Sección 3: SUS ---
  form.addPageBreakItem().setTitle('Cuestionario SUS')
    .setHelpText('Indica tu grado de acuerdo tras usar la herramienta.');
  var sus = [
    'SUS1 — Usaría esta herramienta con frecuencia.',
    'SUS2 — La herramienta es innecesariamente compleja.',
    'SUS3 — La herramienta es fácil de usar.',
    'SUS4 — Necesitaría apoyo técnico para poder usarla.',
    'SUS5 — Las funciones están bien integradas.',
    'SUS6 — La herramienta es demasiado inconsistente.',
    'SUS7 — La mayoría aprendería a usarla con rapidez.',
    'SUS8 — Es incómoda/engorrosa de usar.',
    'SUS9 — Me siento seguro/a usándola.',
    'SUS10 — Necesité aprender muchas cosas antes de empezar.'
  ];
  for (var i = 0; i < sus.length; i++) {
    form.addScaleItem().setTitle(sus[i]).setBounds(1, 5)
      .setLabels(acuerdo[0], acuerdo[1]).setRequired(true);
  }

  // --- Sección 4: Aplicabilidad ---
  form.addPageBreakItem().setTitle('Aplicabilidad');
  var app = [
    'A1 — La integraría en mi flujo de CI/CD o auditoría.',
    'A2 — La priorización de riesgo (ranking) se ajusta a cómo auditaría un clúster.',
    'A3 — El informe es accionable (sé qué revisar y por qué).',
    'A4 — Confiaría en el veredicto para decidir revisar un clúster.'
  ];
  for (var j = 0; j < app.length; j++) {
    form.addScaleItem().setTitle(app[j]).setBounds(1, 5)
      .setLabels(acuerdo[0], acuerdo[1]).setRequired(true);
  }

  // --- Sección 5: Preguntas abiertas ---
  form.addPageBreakItem().setTitle('Comentarios');
  form.addParagraphTextItem().setTitle('¿Qué te ha resultado más útil?');
  form.addParagraphTextItem().setTitle('¿Qué cambiarías o te generó fricción?');
  form.addParagraphTextItem().setTitle('¿En qué caso de uso real lo aplicarías?');
  form.addParagraphTextItem().setTitle('Comentarios adicionales');

  Logger.log('URL para responder: ' + form.getPublishedUrl());
  Logger.log('URL para editar:    ' + form.getEditUrl());
}
