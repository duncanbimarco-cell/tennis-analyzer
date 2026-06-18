// 比赛页面 — 实时战术板（重构版）
// 对齐 stats 页面的 currentMatch.logs 数据格式
const app = getApp();

Page({
  data: {
    // —— 比赛状态 ——
    matchInProgress: false,
    cameraReady: false,
    isRecording: false,

    // —— 比分 ——
    scoreA: 0,
    scoreB: 0,
    playerAName: '我方',
    playerBName: '对方',

    // —— 动作按钮定义 ——
    actionTypes: [
      { key: 'smash_score', label: '杀球得分', icon: '💥', cls: 'action-smash' },
      { key: 'net_score',   label: '网前得分', icon: '🥅', cls: 'action-net' },
      { key: 'out_error',   label: '出界失误', icon: '📤', cls: 'action-out' },
      { key: 'net_error',   label: '下网失误', icon: '📥', cls: 'action-down' }
    ],

    // —— 视频 ——
    latestVideoPath: '',

    // —— 事件日志（最近 20 条） ——
    events: [],

    // —— 最后触发的动作（用于按钮高亮反馈） ——
    lastTriggered: null
  },

  /* ================================================================
     生命周期
     ================================================================ */
  onLoad: function () {
    this._segmentTimer = null;
    this._capturing = false;
    this._processing = false;
    this.cameraCtx = null;

    // 预授权相机 & 录音
    this._requestAuth();
  },

  onUnload: function () {
    this._teardownRecording();
  },

  /* ================================================================
     权限
     ================================================================ */
  _requestAuth: function () {
    wx.authorize({
      scope: 'scope.camera',
      success: () => {
        wx.authorize({
          scope: 'scope.record',
          success: () => {
            this.setData({ cameraReady: true });
          },
          fail: () => {
            // 录音未授权 — 仍可比赛，只是没有视频
            this.setData({ cameraReady: true });
            console.warn('录音权限未授权，视频片段将不可用');
          }
        });
      },
      fail: () => {
        // 相机未授权 — 仍可比赛
        this.setData({ cameraReady: false });
        console.warn('相机权限未授权，视频片段将不可用');
      }
    });
  },

  /* ================================================================
     开始 / 结束比赛
     ================================================================ */
  startMatch: function () {
    // 初始化全局 currentMatch（stats 页面的数据源）
    app.globalData.currentMatch = {
      playerA: this.data.playerAName,
      playerB: this.data.playerBName,
      scoreA: 0,
      scoreB: 0,
      logs: [],
      startTime: this._formatTime(new Date()),
      endTime: null
    };

    this.setData({
      matchInProgress: true,
      scoreA: 0,
      scoreB: 0,
      events: [],
      latestVideoPath: '',
      lastTriggered: null
    });

    // 等 camera 组件渲染后启动分段录制
    setTimeout(() => {
      if (this.data.cameraReady) {
        this.cameraCtx = wx.createCameraContext('matchCamera');
        this._recordSegment();
      }
    }, 600);
  },

  endMatch: function () {
    wx.showModal({
      title: '结束比赛',
      content: `当前比分 ${this.data.playerAName} ${this.data.scoreA} : ${this.data.scoreB} ${this.data.playerBName}\n确定结束并保存吗？`,
      confirmText: '确定',
      cancelText: '继续比赛',
      success: (res) => {
        if (res.confirm) this._finishMatch();
      }
    });
  },

  _finishMatch: function () {
    this._teardownRecording();

    const match = app.globalData.currentMatch;
    if (match) {
      match.scoreA = this.data.scoreA;
      match.scoreB = this.data.scoreB;
      match.endTime = this._formatTime(new Date());
      match.logs = this.data.events;   // events 就是完整 logs
    }

    // 推入历史列表（stats 兼容 matches[index]）
    if (!app.globalData.matches) app.globalData.matches = [];
    app.globalData.matches.push(match);
    app.globalData.currentMatch = null;

    this.setData({ matchInProgress: false });

    wx.showToast({ title: '比赛已保存', icon: 'success', duration: 1200 });
    setTimeout(() => wx.navigateBack(), 1200);
  },

  /* ================================================================
     分段循环录制（≈6 秒一段）
     ================================================================ */
  _recordSegment: function () {
    if (!this.data.matchInProgress || !this.cameraCtx) return;
    if (this._capturing) return;

    this.cameraCtx.startRecord({
      success: () => {
        this.setData({ isRecording: true });
      },
      fail: (err) => {
        console.error('[record] startRecord 失败:', err);
        // 1 秒后重试
        setTimeout(() => this._recordSegment(), 1000);
      }
    });

    // 6 秒后自动切段
    this._segmentTimer = setTimeout(() => {
      this._captureClip((videoPath) => {
        this.setData({ latestVideoPath: videoPath || '' });
        this._recordSegment();   // 下一段
      });
    }, 6000);
  },

  /**
   * 停止当前录制并拿到临时视频路径
   * @param {function} callback  — 接收 (videoPath: string)
   */
  _captureClip: function (callback) {
    if (this._segmentTimer) {
      clearTimeout(this._segmentTimer);
      this._segmentTimer = null;
    }

    if (this._capturing) {
      // 正在停止中，直接返回上次缓存路径
      callback(this.data.latestVideoPath);
      return;
    }

    if (!this.data.isRecording || !this.cameraCtx) {
      callback('');
      return;
    }

    this._capturing = true;

    this.cameraCtx.stopRecord({
      success: (res) => {
        this._capturing = false;
        this.setData({ isRecording: false });
        callback(res.tempVideoPath || '');
      },
      fail: (err) => {
        this._capturing = false;
        this.setData({ isRecording: false });
        console.error('[record] stopRecord 失败:', err);
        callback('');
      }
    });
  },

  _teardownRecording: function () {
    if (this._segmentTimer) {
      clearTimeout(this._segmentTimer);
      this._segmentTimer = null;
    }
    if (this.data.isRecording && this.cameraCtx) {
      this.cameraCtx.stopRecord({
        success: () => {},
        fail: () => {}
      });
    }
    this.setData({ isRecording: false });
  },

  /* ================================================================
     计分逻辑（核心）
     ================================================================ */

  /**
   * 计分规则：
   *  - smash_score / net_score → 触发者得分 +1
   *  - out_error   / net_error → 触发者【对手】得分 +1
   *  log.player 始终记录触发者
   */
  onAction: function (e) {
    if (this._processing) return;           // 防连点
    this._processing = true;

    const { type, player } = e.currentTarget.dataset;

    this._captureClip((videoPath) => {
      this._commitAction(type, player, videoPath || '');
      this._processing = false;
      // 恢复录制
      if (this.data.matchInProgress) {
        this._recordSegment();
      }
    });
  },

  _commitAction: function (type, player, videoPath) {
    let { scoreA, scoreB } = this.data;

    // —— 计分 ——
    if (type === 'smash_score' || type === 'net_score') {
      // 得分动作：谁触发谁得分
      if (player === 'A') scoreA++;
      else scoreB++;
    } else {
      // 失误动作：谁触发，对方得分
      if (player === 'A') scoreB++;
      else scoreA++;
    }

    const log = {
      type: type,
      player: player,
      videoPath: videoPath || '',
      scoreA: scoreA,
      scoreB: scoreB,
      timestamp: this._formatTime(new Date())
    };

    // 追加到事件日志
    const events = [...this.data.events, log];

    // 同步 globalData
    if (app.globalData.currentMatch) {
      app.globalData.currentMatch.logs = events;
      app.globalData.currentMatch.scoreA = scoreA;
      app.globalData.currentMatch.scoreB = scoreB;
    }

    // 触发反馈标识（0.4s 后清除）
    const triggerKey = player + '_' + type;
    this.setData({
      scoreA, scoreB,
      events,
      latestVideoPath: '',
      lastTriggered: triggerKey
    });
    setTimeout(() => {
      if (this.data.lastTriggered === triggerKey) {
        this.setData({ lastTriggered: null });
      }
    }, 400);

    // 触感反馈
    wx.vibrateShort({ type: 'medium' });
  },

  /* ================================================================
     撤销
     ================================================================ */
  undoAction: function () {
    const events = this.data.events;
    if (events.length === 0) return;

    const last = events[events.length - 1];
    const newEvents = events.slice(0, -1);

    // 逆算比分
    let { scoreA, scoreB } = this.data;
    if (last.type === 'smash_score' || last.type === 'net_score') {
      if (last.player === 'A') scoreA--;
      else scoreB--;
    } else {
      if (last.player === 'A') scoreB--;
      else scoreA--;
    }

    // 同步 globalData
    if (app.globalData.currentMatch) {
      app.globalData.currentMatch.logs = newEvents;
      app.globalData.currentMatch.scoreA = scoreA;
      app.globalData.currentMatch.scoreB = scoreB;
    }

    this.setData({ scoreA, scoreB, events: newEvents, lastTriggered: null });
    wx.vibrateShort({ type: 'light' });
  },

  /* ================================================================
     工具函数
     ================================================================ */
  _formatTime: function (date) {
    const y = date.getFullYear();
    const mo = String(date.getMonth() + 1).padStart(2, '0');
    const d = String(date.getDate()).padStart(2, '0');
    const h = String(date.getHours()).padStart(2, '0');
    const mi = String(date.getMinutes()).padStart(2, '0');
    const s = String(date.getSeconds()).padStart(2, '0');
    return `${y}-${mo}-${d} ${h}:${mi}:${s}`;
  }
});
