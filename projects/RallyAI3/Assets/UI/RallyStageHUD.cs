using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;
using Core.Environment;
using Core.ML;
using Core.Physics;

namespace UI
{
    public sealed class RallyStageHUD : MonoBehaviour
    {
        VehicleController vehicle;
        TrackGenerator track;
        Text speedText, gearText, timerText, stageText, modeText;
        SpectatorDirector spectator;
        Image[] rpmSegments;
        Button[] cameraButtons;
        Image[] cameraButtonImages;
        Image progressFill;
        RectTransform progressRect;
        GameObject viewerCompletePanel;
        Text resultTitle, resultStats;
        RallyAgent resultAgent;
        bool wasPaused;
        float[] routeDistance;
        int routeStart, routeFinish;
        float routeLength;
        float startTime;
        Font font;
        int displayedCameraMode = -1;

        static readonly SpectatorDirector.CameraMode[] CameraModes =
        {
            SpectatorDirector.CameraMode.ThirdPerson,
            SpectatorDirector.CameraMode.Helicopter,
            SpectatorDirector.CameraMode.Hood,
            SpectatorDirector.CameraMode.Driver,
            SpectatorDirector.CameraMode.Trackside,
            SpectatorDirector.CameraMode.Cinematic
        };

        static readonly string[] CameraLabels =
        {
            "1  CHASE", "2  HELICOPTER", "3  HOOD", "4  DRIVER", "5  TRACKSIDE", "6  CINEMATIC"
        };

        public void Configure(VehicleController car, TrackGenerator course)
        {
            vehicle = car;
            track = course;
            Camera camera = Camera.main;
            spectator = camera != null ? camera.GetComponent<SpectatorDirector>() : null;
            startTime = Time.time;
            BuildRouteDistance();
            Build();
        }

        void BuildRouteDistance()
        {
            if (track == null || track.waypoints.Count < 2) return;
            routeStart = Mathf.Clamp(track.spawnWaypointIndex, 0, track.waypoints.Count - 2);
            routeFinish = Mathf.Clamp(track.FinishWaypointIndex, routeStart + 1, track.waypoints.Count - 1);
            routeDistance = new float[routeFinish - routeStart + 1];
            for (int i = routeStart + 1; i <= routeFinish; i++)
            {
                Vector3 delta = track.waypoints[i] - track.waypoints[i - 1];
                routeDistance[i - routeStart] = routeDistance[i - routeStart - 1] +
                                                new Vector2(delta.x, delta.z).magnitude;
            }
            routeLength = routeDistance[routeDistance.Length - 1];
        }

        void Build()
        {
            font = Font.CreateDynamicFontFromOSFont(new[] { "Arial", "Helvetica Neue", "Menlo" }, 48);
            var canvas = gameObject.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = 50;
            var scaler = gameObject.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1600f, 900f);
            scaler.matchWidthOrHeight = 0.5f;
            gameObject.AddComponent<GraphicRaycaster>();
            EnsureEventSystem();

            RectTransform topPanel = Panel("Stage Label", new Vector2(0f, 1f), new Vector2(0f, 1f),
                                           new Vector2(28f, -28f), new Vector2(380f, 90f), new Color(0.025f, 0.035f, 0.04f, 0.76f));
            LabLaunch config = LabRuntime.Config;
            string family = config != null && !string.IsNullOrEmpty(config.courseFamily)
                ? config.courseFamily.ToUpperInvariant() + (config.courseFamily=="nordschleife"?" ASPHALT":" GRAVEL") : "GRAVEL STAGE";
            Label(topPanel, "RALLY STAGE  /  " + family, 18, FontStyle.Bold,
                  new Vector2(18f, -10f), new Vector2(344f, 28f), TextAnchor.MiddleLeft, Color.white);
            string identity = config != null && !string.IsNullOrEmpty(config.courseId)
                ? "  /  " + ShortId(config.courseId) : string.Empty;
            string seed = track != null ? track.CurrentSeed.ToString() : "UNKNOWN";
            stageText = Label(topPanel, "SEED " + seed + identity, 12, FontStyle.Bold,
                              new Vector2(18f, -39f), new Vector2(344f, 18f), TextAnchor.MiddleLeft, new Color(0.78f, 0.82f, 0.78f));
            modeText = Label(topPanel, "LIVE DRIVE", 11, FontStyle.Bold,
                             new Vector2(18f, -62f), new Vector2(344f, 16f), TextAnchor.MiddleLeft, new Color(0.96f, 0.73f, 0.20f));
            RectTransform timerPanel = Panel("Stage Clock", new Vector2(1f, 1f), new Vector2(1f, 1f),
                                             new Vector2(-28f, -28f), new Vector2(220f, 66f), new Color(0.025f, 0.035f, 0.04f, 0.76f));
            Label(timerPanel, "STAGE TIME", 10, FontStyle.Bold, new Vector2(14f, -5f), new Vector2(192f, 16f),
                  TextAnchor.MiddleCenter, new Color(0.78f, 0.82f, 0.78f));
            timerText = Label(timerPanel, "00:00.00", 25, FontStyle.Bold, new Vector2(0f, -18f), new Vector2(200f, 42f), TextAnchor.MiddleCenter, Color.white);

            RectTransform progressPanel = Panel("Stage Progress", new Vector2(0.5f, 1f), new Vector2(0.5f, 1f),
                                                new Vector2(0f, -29f), new Vector2(520f, 42f), new Color(0.025f, 0.035f, 0.04f, 0.68f));
            Label(progressPanel, "STAGE DISTANCE", 12, FontStyle.Bold, new Vector2(12f, -3f), new Vector2(150f, 18f), TextAnchor.MiddleLeft, new Color(0.78f, 0.85f, 0.82f));
            Label(progressPanel, track != null ? Mathf.RoundToInt(track.trackLength).ToString("N0") + " M" : "", 12, FontStyle.Bold,
                  new Vector2(410f, -3f), new Vector2(94f, 18f), TextAnchor.MiddleRight, new Color(0.78f, 0.85f, 0.82f));
            var rail = new GameObject("Distance Rail", typeof(RectTransform), typeof(Image));
            rail.transform.SetParent(progressPanel, false);
            RectTransform railRect = rail.GetComponent<RectTransform>();
            railRect.anchorMin = new Vector2(0f, 0f);
            railRect.anchorMax = new Vector2(1f, 0f);
            railRect.pivot = new Vector2(0.5f, 0f);
            railRect.anchoredPosition = new Vector2(0f, 7f);
            railRect.sizeDelta = new Vector2(-24f, 6f);
            rail.GetComponent<Image>().color = new Color(0.18f, 0.24f, 0.23f, 1f);
            var fill = new GameObject("Distance Complete", typeof(RectTransform), typeof(Image));
            fill.transform.SetParent(rail.transform, false);
            progressRect = fill.GetComponent<RectTransform>();
            progressRect.anchorMin = new Vector2(0f, 0f);
            progressRect.anchorMax = new Vector2(0f, 1f);
            progressRect.pivot = new Vector2(0f, 0.5f);
            progressRect.anchoredPosition = Vector2.zero;
            progressRect.sizeDelta = Vector2.zero;
            progressFill = fill.GetComponent<Image>();
            progressFill.color = new Color(0.96f, 0.73f, 0.18f, 1f);
            progressFill.type = Image.Type.Simple;

            RectTransform carPanel = Panel("Car Readout", new Vector2(1f, 0f), new Vector2(1f, 0f),
                                            new Vector2(-28f, 28f), new Vector2(292f, 142f), new Color(0.025f, 0.035f, 0.04f, 0.82f));
            speedText = Label(carPanel, "000", 60, FontStyle.Bold, new Vector2(14f, -7f), new Vector2(160f, 76f), TextAnchor.MiddleLeft, Color.white);
            Label(carPanel, "MPH", 13, FontStyle.Bold, new Vector2(177f, -16f), new Vector2(88f, 26f), TextAnchor.MiddleLeft, new Color(0.78f, 0.82f, 0.78f));
            gearText = Label(carPanel, "1", 28, FontStyle.Bold, new Vector2(177f, -43f), new Vector2(82f, 38f), TextAnchor.MiddleLeft, new Color(0.98f, 0.76f, 0.18f));
            Label(carPanel, "GEAR", 9, FontStyle.Bold, new Vector2(218f, -50f), new Vector2(54f, 24f), TextAnchor.MiddleLeft, new Color(0.65f, 0.71f, 0.68f));
            rpmSegments = new Image[12];
            for (int i = 0; i < rpmSegments.Length; i++)
            {
                var segment = new GameObject("RPM " + (i + 1), typeof(RectTransform), typeof(Image));
                segment.transform.SetParent(carPanel, false);
                var rect = segment.GetComponent<RectTransform>();
                rect.anchorMin = new Vector2(0f, 0f);
                rect.anchorMax = new Vector2(0f, 0f);
                rect.pivot = new Vector2(0f, 0f);
                rect.anchoredPosition = new Vector2(16f + i * 21.5f, 15f);
                rect.sizeDelta = new Vector2(16f, 9f);
                rpmSegments[i] = segment.GetComponent<Image>();
                rpmSegments[i].color = new Color(0.18f, 0.24f, 0.23f, 1f);
                rpmSegments[i].raycastTarget = false;
            }
            Label(carPanel, "RPM", 9, FontStyle.Bold, new Vector2(16f, -107f), new Vector2(44f, 18f), TextAnchor.MiddleLeft, new Color(0.65f, 0.71f, 0.68f));
            modeText.text = SessionLabel();
            BuildCameraSelector();
            BuildViewerStatus();
        }

        void BuildViewerStatus()
        {
            if (LabRuntime.Config == null || !LabRuntime.Config.viewer) return;
            resultAgent=vehicle.GetComponent<RallyAgent>();
            RectTransform panel = Panel("Viewer Attempt Complete", new Vector2(0f, 0f), new Vector2(0f, 0f),
                                        new Vector2(28f, 28f), new Vector2(390f, 262f),
                                        new Color(0.025f, 0.035f, 0.04f, 0.90f));
            resultTitle=Label(panel, "ATTEMPT COMPLETE", 24, FontStyle.Bold,
                  new Vector2(16f, -14f), new Vector2(360f, 34f), TextAnchor.MiddleLeft,
                  new Color(1f, 0.83f, 0.30f));
            resultStats=Label(panel,"",15,FontStyle.Normal,new Vector2(16,-55),new Vector2(360,145),TextAnchor.UpperLeft,Color.white);
            ResultButton(panel,"RETRY",new Vector2(16,-213),()=>{LabRuntime.RetryViewer();startTime=Time.time;});
            ResultButton(panel,"EXIT",new Vector2(204,-213),()=>Application.Quit());
            viewerCompletePanel = panel.gameObject;
            viewerCompletePanel.SetActive(false);
        }

        void ResultButton(Transform parent,string title,Vector2 position,UnityEngine.Events.UnityAction action)
        {
            var go=new GameObject(title,typeof(RectTransform),typeof(Image),typeof(Button));
            go.transform.SetParent(parent,false);
            var rect=go.GetComponent<RectTransform>(); rect.anchorMin=rect.anchorMax=rect.pivot=new Vector2(0,1);
            rect.anchoredPosition=position; rect.sizeDelta=new Vector2(170,34);
            go.GetComponent<Image>().color=new Color(.12f,.18f,.17f);
            go.GetComponent<Button>().onClick.AddListener(action);
            Label(go.transform,title,14,FontStyle.Bold,Vector2.zero,new Vector2(170,34),TextAnchor.MiddleCenter,Color.white);
        }

        void RefreshResult()
        {
            if(resultAgent==null || !resultAgent.ViewerResult.HasValue) return;
            var result=resultAgent.ViewerResult.Value;
            resultTitle.text=result.valid?"STAGE COMPLETE":result.outcome.ToString().ToUpperInvariant();
            string time=result.valid?$"TIME  {result.seconds:0.00} s":$"NO VALID TIME  /  ELAPSED {result.seconds:0.00} s";
            string split=result.splits!=null && result.splits.Length>1?$"FINAL SECTOR  {result.splits[result.splits.Length-1]-result.splits[result.splits.Length-2]:0.00} s":"FINAL SECTOR  --";
            resultStats.text=$"{time}\nPEAK SPEED  {result.peakSpeed*2.2369363f:0} MPH\nGATES  {result.waypointsReached}/{result.waypointTarget}  /  ATTEMPT {result.attempt}\n{split}\nWATCH INFERENCE  /  {ShortId(LabRuntime.Config.checkpointId)}";
        }

        void BuildCameraSelector()
        {
            RectTransform selector = Panel("Camera Selector", new Vector2(0.5f, 1f), new Vector2(0.5f, 1f),
                                           new Vector2(0f, -78f), new Vector2(600f, 38f),
                                           new Color(0.025f, 0.035f, 0.04f, 0.72f));
            Label(selector, "VIEW", 10, FontStyle.Bold, new Vector2(10f, -10f), new Vector2(34f, 17f),
                  TextAnchor.MiddleLeft, new Color(0.80f, 0.84f, 0.81f));
            cameraButtons = new Button[CameraModes.Length];
            cameraButtonImages = new Image[CameraModes.Length];
            for (int i = 0; i < CameraModes.Length; i++)
            {
                CameraModeButton(selector, i);
            }
            RefreshCameraButtons(true);
        }

        void CameraModeButton(Transform parent, int index)
        {
            var go = new GameObject("Camera " + CameraLabels[index], typeof(RectTransform), typeof(Image), typeof(Button));
            go.transform.SetParent(parent, false);
            RectTransform rect = go.GetComponent<RectTransform>();
            rect.anchorMin = new Vector2(0f, 1f);
            rect.anchorMax = new Vector2(0f, 1f);
            rect.pivot = new Vector2(0f, 1f);
            rect.anchoredPosition = new Vector2(50f + index * 88f, -6f);
            rect.sizeDelta = new Vector2(86f, 26f);

            Image image = go.GetComponent<Image>();
            image.color = new Color(0.11f, 0.15f, 0.16f, 0.96f);
            Button button = go.GetComponent<Button>();
            button.targetGraphic = image;
            button.transition = Selectable.Transition.None;
            ColorBlock colors = button.colors;
            colors.normalColor = new Color(0.15f, 0.19f, 0.19f, 1f);
            colors.highlightedColor = new Color(0.32f, 0.37f, 0.33f, 1f);
            colors.pressedColor = new Color(0.83f, 0.61f, 0.16f, 1f);
            colors.selectedColor = colors.highlightedColor;
            button.colors = colors;

            var labelObject = new GameObject("Mode Label", typeof(RectTransform), typeof(Text));
            labelObject.transform.SetParent(go.transform, false);
            RectTransform labelRect = labelObject.GetComponent<RectTransform>();
            labelRect.anchorMin = Vector2.zero;
            labelRect.anchorMax = Vector2.one;
            labelRect.offsetMin = new Vector2(2f, 0f);
            labelRect.offsetMax = new Vector2(-2f, 0f);
            Text label = labelObject.GetComponent<Text>();
            label.text = CameraLabels[index];
            label.font = font;
            label.fontSize = 11;
            label.fontStyle = FontStyle.Bold;
            label.alignment = TextAnchor.MiddleCenter;
            label.color = Color.white;
            label.raycastTarget = false;
            label.horizontalOverflow = HorizontalWrapMode.Overflow;
            label.verticalOverflow = VerticalWrapMode.Truncate;

            SpectatorDirector.CameraMode selectedMode = CameraModes[index];
            button.onClick.AddListener(() =>
            {
                if (spectator != null) spectator.SetMode(selectedMode);
                RefreshCameraButtons(true);
            });
            cameraButtons[index] = button;
            cameraButtonImages[index] = image;
        }

        static void EnsureEventSystem()
        {
            EventSystem events = FindObjectOfType<EventSystem>();
            if (events == null)
                events = new GameObject("Rally UI Event System", typeof(EventSystem), typeof(StandaloneInputModule))
                    .GetComponent<EventSystem>();
            else if (events.GetComponent<BaseInputModule>() == null)
                events.gameObject.AddComponent<StandaloneInputModule>();
        }

        void RefreshCameraButtons(bool force)
        {
            if (spectator == null)
            {
                Camera camera = Camera.main;
                spectator = camera != null ? camera.GetComponent<SpectatorDirector>() : null;
            }
            if (spectator == null || cameraButtons == null) return;
            int selected = System.Array.IndexOf(CameraModes, spectator.currentMode);
            if (!force && displayedCameraMode == selected) return;
            displayedCameraMode = selected;
            for (int i = 0; i < cameraButtons.Length; i++)
            {
                bool active = i == selected;
                cameraButtonImages[i].color = active
                    ? new Color(0.12f, 0.43f, 0.34f, 1f)
                    : new Color(0.11f, 0.15f, 0.16f, 0.96f);
            }
        }

        string SessionLabel()
        {
            GhostPlayer replay = FindObjectOfType<GhostPlayer>();
            if (replay != null && replay.isActiveAndEnabled && replay.Path != null)
                return "RECORDED REPLAY  /  " + replay.Path.label;

            LabLaunch config = LabRuntime.Config;
            if (config == null) return "LIVE TRAINING DRIVE";
            if (Core.ML.LabRuntime.CircuitInspection) return "CAMERA INSPECTION  /  UNRANKED";
            if (config.controlProbe) return "DIAGNOSTIC DRIVE  /  UNRANKED";
            if (config.viewer) return "LIVE POLICY  /  " + ShortId(config.checkpointId);
            if (config.evaluation) return "EVALUATION  /  " + ShortId(config.checkpointId);
            return "LIVE TRAINING DRIVE  /  " + config.runId;
        }

        static string ShortId(string value)
            => string.IsNullOrEmpty(value) ? "UNKNOWN" : value.Substring(0, Mathf.Min(8, value.Length)).ToUpperInvariant();

        void Update()
        {
            RefreshCameraButtons(false);
            if (viewerCompletePanel != null && viewerCompletePanel.activeSelf != LabRuntime.ViewerPaused)
                viewerCompletePanel.SetActive(LabRuntime.ViewerPaused);
            if(LabRuntime.ViewerPaused && !wasPaused) RefreshResult();
            wasPaused=LabRuntime.ViewerPaused;
            if(cameraButtons!=null) foreach(var button in cameraButtons) button.interactable=!LabRuntime.ViewerPaused;
            if (vehicle == null) return;
            float seconds = Mathf.Max(0f, Time.time - startTime);
            if(LabRuntime.ViewerPaused && resultAgent!=null && resultAgent.ViewerResult.HasValue)
                seconds=resultAgent.ViewerResult.Value.seconds;
            int minutes = Mathf.FloorToInt(seconds / 60f);
            timerText.text = minutes.ToString("00") + ":" + (seconds % 60f).ToString("00.00");
            speedText.text = Mathf.RoundToInt(Mathf.Abs(vehicle.currentSpeedKmh) / 1.609344f).ToString("000");
            gearText.text = vehicle.currentGear < 0 ? "R" : vehicle.currentGear == 0 ? "N" : vehicle.currentGear.ToString();
            float rpm = Mathf.Clamp01(vehicle.NormalisedRpm);
            int lit = Mathf.RoundToInt(rpm * rpmSegments.Length);
            for (int i = 0; i < rpmSegments.Length; i++)
            {
                Color active = i >= 10 ? new Color(0.91f, 0.22f, 0.12f) : i >= 8 ? new Color(0.96f, 0.69f, 0.18f) : new Color(0.36f, 0.78f, 0.42f);
                rpmSegments[i].color = i < lit ? active : new Color(0.18f, 0.24f, 0.23f, 1f);
            }
            if (progressFill != null && track != null)
            {
                float bestDistanceSq = float.MaxValue;
                float progressDistance = 0f;
                if (routeDistance != null && routeLength > 0f)
                {
                    Vector3 car = vehicle.transform.position;
                    for (int i = routeStart; i < routeFinish; i++)
                    {
                        Vector3 a = track.transform.TransformPoint(track.waypoints[i]);
                        Vector3 b = track.transform.TransformPoint(track.waypoints[i + 1]);
                        Vector2 span = new Vector2(b.x - a.x, b.z - a.z);
                        Vector2 relative = new Vector2(car.x - a.x, car.z - a.z);
                        float t = Mathf.Clamp01(Vector2.Dot(relative, span) / Mathf.Max(0.001f, span.sqrMagnitude));
                        Vector2 nearest = new Vector2(a.x, a.z) + span * t;
                        float distanceSq = (new Vector2(car.x, car.z) - nearest).sqrMagnitude;
                        if (distanceSq >= bestDistanceSq) continue;
                        bestDistanceSq = distanceSq;
                        int segment = i - routeStart;
                        progressDistance = routeDistance[segment] + span.magnitude * t;
                    }
                }
                float fraction = routeLength > 0f ? Mathf.Clamp01(progressDistance / routeLength) : 0f;
                progressRect.sizeDelta = new Vector2(496f * fraction, 0f);
            }
        }

        RectTransform Panel(string name, Vector2 anchorMin, Vector2 anchorMax,
                             Vector2 offset, Vector2 size, Color color)
        {
            var go = new GameObject(name, typeof(RectTransform), typeof(Image));
            go.transform.SetParent(transform, false);
            var rect = go.GetComponent<RectTransform>();
            rect.anchorMin = anchorMin;
            rect.anchorMax = anchorMax;
            rect.pivot = anchorMin == Vector2.zero ? Vector2.zero : anchorMax;
            rect.anchoredPosition = offset;
            rect.sizeDelta = size;
            var image = go.GetComponent<Image>();
            image.color = color;
            image.raycastTarget = false;
            var accent = new GameObject("Stage Gold Edge", typeof(RectTransform), typeof(Image));
            accent.transform.SetParent(go.transform, false);
            var accentRect = accent.GetComponent<RectTransform>();
            accentRect.anchorMin = new Vector2(0f, 1f);
            accentRect.anchorMax = new Vector2(1f, 1f);
            accentRect.pivot = new Vector2(0.5f, 1f);
            accentRect.anchoredPosition = Vector2.zero;
            accentRect.sizeDelta = new Vector2(0f, 2f);
            var accentImage = accent.GetComponent<Image>();
            accentImage.color = new Color(0.98f, 0.72f, 0.16f, 0.92f);
            accentImage.raycastTarget = false;
            return rect;
        }

        Text Label(Transform parent, string value, int size, FontStyle style,
                   Vector2 offset, Vector2 dimensions, TextAnchor alignment, Color color)
        {
            var go = new GameObject("Label", typeof(RectTransform), typeof(Text));
            go.transform.SetParent(parent, false);
            var rect = go.GetComponent<RectTransform>();
            rect.anchorMin = new Vector2(0f, 1f);
            rect.anchorMax = new Vector2(0f, 1f);
            rect.pivot = new Vector2(0f, 1f);
            rect.anchoredPosition = offset;
            rect.sizeDelta = dimensions;
            var text = go.GetComponent<Text>();
            text.text = value;
            text.font = font;
            text.fontSize = size;
            text.fontStyle = style;
            text.color = color;
            text.alignment = alignment;
            text.raycastTarget = false;
            text.horizontalOverflow = HorizontalWrapMode.Overflow;
            text.verticalOverflow = VerticalWrapMode.Overflow;
            return text;
        }
    }
}
