(function() {
    var UNICODE = {
        wK: '\u2654', wQ: '\u2655', wR: '\u2656', wB: '\u2657', wN: '\u2658', wP: '\u2659',
        bK: '\u265A', bQ: '\u265B', bR: '\u265C', bB: '\u265D', bN: '\u265E', bP: '\u265F'
    };

    var FILES = ['a','b','c','d','e','f','g','h'];
    var INIT_POS = [
        ['bR','bN','bB','bQ','bK','bB','bN','bR'],
        ['bP','bP','bP','bP','bP','bP','bP','bP'],
        [null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null],
        ['wP','wP','wP','wP','wP','wP','wP','wP'],
        ['wR','wN','wB','wQ','wK','wB','wN','wR']
    ];

    // Пользователь и игровые данные для датасета
    var userId = localStorage.getItem('sfedu_user_id') || 'user_' + Math.random().toString(36).substr(2, 9);
    localStorage.setItem('sfedu_user_id', userId);
    var gameId = 'game_' + Date.now();
    var moveCount = 0;

    var position = [];
    var selected = null;
    var lastFrom = null;
    var lastTo = null;
    var turn = 'w';
    var moveHistory = [];
    var moveNumber = 1;
    var castling = { wK: true, wQ: true, bK: true, bQ: true };
    var enPassant = null;
    var capturedByWhite = [];
    var capturedByBlack = [];
    var boardEl = document.getElementById('board');
    var lastMove = null;
    var recommendedMove = null; // Рекомендация AI (например "e2e4")

    function init() {
        position = [];
        for (var r = 0; r < 8; r++) {
            position[r] = [];
            for (var c = 0; c < 8; c++) {
                position[r][c] = INIT_POS[r][c];
            }
        }
        turn = 'w';
        selected = null;
        lastFrom = null;
        lastTo = null;
        lastMove = null;
        moveHistory = [];
        moveNumber = 1;
        castling = { wK: true, wQ: true, bK: true, bQ: true };
        enPassant = null;
        capturedByWhite = [];
        capturedByBlack = [];
        halfmoveClock = 0;
        render();
        updateStatus();
        updateMaterialDisplay();
        positionSnapshots = []; viewIndex = -1; isViewMode = false;
        takeSnapshot();
        updateNavButtons();
    }

    function toFEN() {
        var fen = '';
        for (var r = 0; r < 8; r++) {
            var empty = 0;
            for (var c = 0; c < 8; c++) {
                var p = position[r][c];
                if (!p) { empty++; continue; }
                if (empty > 0) { fen += empty; empty = 0; }
                var map = {wP:'P',wN:'N',wB:'B',wR:'R',wQ:'Q',wK:'K',bP:'p',bN:'n',bB:'b',bR:'r',bQ:'q',bK:'k'};
                fen += map[p] || '?';
            }
            if (empty > 0) fen += empty;
            if (r < 7) fen += '/';
        }
        fen += ' ' + (turn === 'w' ? 'w' : 'b');
        var c_str = '';
        if (castling.wK) c_str += 'K';
        if (castling.wQ) c_str += 'Q';
        if (castling.bK) c_str += 'k';
        if (castling.bQ) c_str += 'q';
        fen += ' ' + (c_str || '-');
        fen += ' ' + (enPassant || '-');
        fen += ' ' + halfmoveClock;
        fen += ' ' + moveNumber;
        return fen;
    }

    function sqName(r, c) { return FILES[c] + (8 - r); }
    function rc(name) { return { r: 8 - parseInt(name[1]), c: FILES.indexOf(name[0]) }; }

    function at(r, c) {
        if (r < 0 || r > 7 || c < 0 || c > 7) return undefined;
        return position[r][c];
    }

    function colorOf(p) { return p ? p[0] : null; }
    function typeOf(p) { return p ? p[1] : null; }
    function enemy() { return turn === 'w' ? 'b' : 'w'; }

    var PIECE_VALUES = { P: 1, N: 3, B: 3, R: 5, Q: 9, K: 0 };

    function getMaterial() {
        var white = 0, black = 0;
        for (var r = 0; r < 8; r++) {
            for (var c = 0; c < 8; c++) {
                var p = position[r][c];
                if (!p) continue;
                var val = PIECE_VALUES[typeOf(p)];
                if (colorOf(p) === 'w') white += val;
                else black += val;
            }
        }
        return { white: white, black: black, diff: white - black };
    }

    function updateMaterialDisplay() {
        var m = getMaterial();
        var topEl = document.getElementById('material-display-top');
        var bottomEl = document.getElementById('material-display-bottom');
        var diff = m.diff;

        function getCapturedHtml(captures) {
            var CAPTURED_ICONS = {
                'bP': '\u265F', 'bN': '\u265E', 'bB': '\u265D', 'bR': '\u265C', 'bQ': '\u265B', 'bK': '\u265A',
                'wP': '\u2659', 'wN': '\u2658', 'wB': '\u2657', 'wR': '\u2656', 'wQ': '\u2655', 'wK': '\u2654'
            };
            var html = '';
            for (var i = 0; i < captures.length; i++) {
                var p = captures[i];
                var icon = CAPTURED_ICONS[p] || '';
                html += '<span style="font-size:16px;color:#fff;text-shadow:-1px -1px 0 #000,1px -1px 0 #000,-1px 1px 0 #000,1px 1px 0 #000">' + icon + '</span>';
            }
            return html;
        }

        var whiteCapturesHtml = getCapturedHtml(capturedByWhite);
        var blackCapturesHtml = getCapturedHtml(capturedByBlack);

        if (diff === 0) {
            topEl.className = 'material-display equal';
            topEl.innerHTML = blackCapturesHtml;
            bottomEl.className = 'material-display equal';
            bottomEl.innerHTML = whiteCapturesHtml;
        } else if (diff > 0) {
            topEl.className = 'material-display equal';
            topEl.innerHTML = blackCapturesHtml;
            bottomEl.className = 'material-display white-adv';
            bottomEl.innerHTML = '<span style="font-weight:bold;">+' + diff + '</span> ' + whiteCapturesHtml;
        } else {
            topEl.className = 'material-display white-adv';
            topEl.innerHTML = '<span style="font-weight:bold;">+' + Math.abs(diff) + '</span> ' + blackCapturesHtml;
            bottomEl.className = 'material-display equal';
            bottomEl.innerHTML = whiteCapturesHtml;
        }
    }

    function legalMoves(fromR, fromC) {
        var piece = at(fromR, fromC);
        if (!piece || colorOf(piece) !== turn) return [];
        var raw = rawMoves(fromR, fromC, piece);
        var legal = [];
        for (var i = 0; i < raw.length; i++) {
            var m = raw[i];
            if (!wouldBeInCheck(fromR, fromC, m.r, m.c, m.special)) {
                legal.push(m);
            }
        }
        return legal;
    }

    function rawMoves(fr, fc, piece) {
        var moves = [];
        var color = colorOf(piece);
        var type = typeOf(piece);
        var opp = color === 'w' ? 'b' : 'w';

        if (type === 'P') {
            var dir = color === 'w' ? -1 : 1;
            var startRow = color === 'w' ? 6 : 1;
            if (!at(fr + dir, fc)) {
                moves.push({ r: fr + dir, c: fc });
                if (fr === startRow && !at(fr + 2 * dir, fc)) {
                    moves.push({ r: fr + 2 * dir, c: fc, special: 'double' });
                }
            }
            for (var dc = -1; dc <= 1; dc += 2) {
                var nc = fc + dc;
                if (nc < 0 || nc > 7) continue;
                var target = at(fr + dir, nc);
                if (target && colorOf(target) === opp) {
                    moves.push({ r: fr + dir, c: nc });
                }
                if (enPassant && enPassant === sqName(fr + dir, nc)) {
                    moves.push({ r: fr + dir, c: nc, special: 'enpassant' });
                }
            }
        }

        if (type === 'N') {
            var jumps = [[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]];
            for (var j = 0; j < jumps.length; j++) {
                var nr = fr + jumps[j][0], nc2 = fc + jumps[j][1];
                if (nr < 0 || nr > 7 || nc2 < 0 || nc2 > 7) continue;
                var t = at(nr, nc2);
                if (!t || colorOf(t) === opp) moves.push({ r: nr, c: nc2 });
            }
        }

        if (type === 'B' || type === 'Q') {
            var bDirs = [[-1,-1],[-1,1],[1,-1],[1,1]];
            for (var d = 0; d < bDirs.length; d++) slide(fr, fc, bDirs[d], opp, moves);
        }

        if (type === 'R' || type === 'Q') {
            var rDirs = [[-1,0],[1,0],[0,-1],[0,1]];
            for (var d2 = 0; d2 < rDirs.length; d2++) slide(fr, fc, rDirs[d2], opp, moves);
        }

        if (type === 'K') {
            for (var dr = -1; dr <= 1; dr++) {
                for (var dc3 = -1; dc3 <= 1; dc3++) {
                    if (dr === 0 && dc3 === 0) continue;
                    var kr = fr + dr, kc = fc + dc3;
                    if (kr < 0 || kr > 7 || kc < 0 || kc > 7) continue;
                    var kt = at(kr, kc);
                    if (!kt || colorOf(kt) === opp) moves.push({ r: kr, c: kc });
                }
            }
            if (castling[color + 'K'] && !at(fr, 5) && !at(fr, 6) && at(fr, 7) === color + 'R') {
                if (!isAttacked(fr, 4, opp) && !isAttacked(fr, 5, opp) && !isAttacked(fr, 6, opp)) {
                    moves.push({ r: fr, c: 6, special: 'castle-k' });
                }
            }
            if (castling[color + 'Q'] && !at(fr, 3) && !at(fr, 2) && !at(fr, 1) && at(fr, 0) === color + 'R') {
                if (!isAttacked(fr, 4, opp) && !isAttacked(fr, 3, opp) && !isAttacked(fr, 2, opp)) {
                    moves.push({ r: fr, c: 2, special: 'castle-q' });
                }
            }
        }

        return moves;
    }

    function slide(fr, fc, dir, opp, moves) {
        var r = fr + dir[0], c = fc + dir[1];
        while (r >= 0 && r <= 7 && c >= 0 && c <= 7) {
            var t = at(r, c);
            if (!t) { moves.push({ r: r, c: c }); }
            else if (colorOf(t) === opp) { moves.push({ r: r, c: c }); break; }
            else break;
            r += dir[0]; c += dir[1];
        }
    }

    function isAttacked(r, c, byColor) {
        for (var rr = 0; rr < 8; rr++) {
            for (var cc = 0; cc < 8; cc++) {
                var p = at(rr, cc);
                if (!p || colorOf(p) !== byColor) continue;
                var mvs = rawMoves(rr, cc, p);
                for (var i = 0; i < mvs.length; i++) {
                    if (mvs[i].r === r && mvs[i].c === c) return true;
                }
            }
        }
        return false;
    }

    function findKing(color) {
        for (var r = 0; r < 8; r++)
            for (var c = 0; c < 8; c++)
                if (position[r][c] === color + 'K') return { r: r, c: c };
        return null;
    }

    function inCheck(color) {
        var k = findKing(color);
        if (!k) return false;
        return isAttacked(k.r, k.c, color === 'w' ? 'b' : 'w');
    }

    function wouldBeInCheck(fr, fc, tr, tc, special) {
        var backup = [];
        backup.push({ r: fr, c: fc, v: position[fr][fc] });
        backup.push({ r: tr, c: tc, v: position[tr][tc] });

        var movingPiece = position[fr][fc];
        position[tr][tc] = movingPiece;
        position[fr][fc] = null;

        if (special === 'enpassant') {
            var epR = colorOf(movingPiece) === 'w' ? tr + 1 : tr - 1;
            backup.push({ r: epR, c: tc, v: position[epR][tc] });
            position[epR][tc] = null;
        }
        if (special === 'castle-k') {
            backup.push({ r: fr, c: 7, v: position[fr][7] });
            backup.push({ r: fr, c: 5, v: position[fr][5] });
            position[fr][5] = position[fr][7];
            position[fr][7] = null;
        }
        if (special === 'castle-q') {
            backup.push({ r: fr, c: 0, v: position[fr][0] });
            backup.push({ r: fr, c: 3, v: position[fr][3] });
            position[fr][3] = position[fr][0];
            position[fr][0] = null;
        }

        var check = inCheck(colorOf(movingPiece));

        for (var i = 0; i < backup.length; i++) {
            position[backup[i].r][backup[i].c] = backup[i].v;
        }

        return check;
    }

    function hasAnyLegalMove(color) {
        var savedTurn = turn;
        turn = color;
        for (var r = 0; r < 8; r++) {
            for (var c = 0; c < 8; c++) {
                if (at(r, c) && colorOf(at(r, c)) === color) {
                    if (legalMoves(r, c).length > 0) { turn = savedTurn; return true; }
                }
            }
        }
        turn = savedTurn;
        return false;
    }

    function makeMove(fr, fc, tr, tc, special) {
        var piece = position[fr][fc];
        var captured = position[tr][tc];
        var color = colorOf(piece);
        var type = typeOf(piece);
        var moveStr = '';

        if (special === 'castle-k') { moveStr = 'O-O'; }
        else if (special === 'castle-q') { moveStr = 'O-O-O'; }
        else {
            if (type !== 'P') moveStr += type;
            if (captured || special === 'enpassant') {
                if (type === 'P') moveStr += FILES[fc];
                moveStr += 'x';
            }
            moveStr += sqName(tr, tc);
        }

        lastMove = { fromR: fr, fromC: fc, toR: tr, toC: tc, piece: piece, special: special };

        position[tr][tc] = piece;
        position[fr][fc] = null;

        if (captured) {
            if (color === 'w') capturedByWhite.push(captured);
            else capturedByBlack.push(captured);
        }

        if (special === 'enpassant') {
            var epRow = color === 'w' ? tr + 1 : tr - 1;
            var epCaptured = color === 'w' ? 'bP' : 'wP';
            if (color === 'w') capturedByWhite.push(epCaptured);
            else capturedByBlack.push(epCaptured);
            position[epRow][tc] = null;
        }
        if (special === 'castle-k') {
            position[fr][5] = position[fr][7];
            position[fr][7] = null;
        }
        if (special === 'castle-q') {
            position[fr][3] = position[fr][0];
            position[fr][0] = null;
        }

        if (special === 'double') {
            enPassant = sqName((fr + tr) / 2, fc);
        } else {
            enPassant = null;
        }

        var promoRow = color === 'w' ? 0 : 7;
        if (type === 'P' && tr === promoRow) {
            position[tr][tc] = color + 'Q';
            moveStr += '=Q';
        }

        if (type === 'K') { castling[color + 'K'] = false; castling[color + 'Q'] = false; }
        if (type === 'R' && fc === 0) castling[color + 'Q'] = false;
        if (type === 'R' && fc === 7) castling[color + 'K'] = false;
        if (tr === 0 && tc === 7) castling['bK'] = false;
        if (tr === 0 && tc === 0) castling['bQ'] = false;
        if (tr === 7 && tc === 7) castling['wK'] = false;
        if (tr === 7 && tc === 0) castling['wQ'] = false;

        if (type === 'P' || captured || special === 'enpassant') {
            halfmoveClock = 0;
        } else {
            halfmoveClock++;
        }

        lastFrom = sqName(fr, fc);
        lastTo = sqName(tr, tc);

        turn = enemy();

        if (inCheck(turn)) moveStr += hasAnyLegalMove(turn) ? '+' : '#';

        if (color === 'w') {
            moveHistory.push({ num: moveNumber, w: moveStr, b: null });
        } else {
            if (moveHistory.length > 0) {
                moveHistory[moveHistory.length - 1].b = moveStr;
            }
            moveNumber++;
        }
        
        // === СОХРАНЕНИЕ В ДАТАСЕТ ===
        // Сохраняем только ходы пользователя (белые)
        if (color === 'w') {
            moveCount++;
            saveMoveToDataset(fr, fc, tr, tc);
        }
        takeSnapshot();
        updateNavButtons();
    }

    // Функция сохранения хода в датасет
    function saveMoveToDataset(fr, fc, tr, tc) {
        // Конвертируем координаты в UCI нотацию (например "e2e4")
        var fromSq = FILES[fc] + (8 - fr);
        var toSq = FILES[tc] + (8 - tr);
        var moveUci = fromSq + toSq;
        
        // Получаем текущий FEN
        var fen = toFEN();
        
        // Отправляем на сервер (асинхронно, без ожидания ответа)
        fetch('/api/save-move-to-dataset', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                fen: fen,
                move: moveUci,
                user_id: userId,
                game_id: gameId
            })
        }).catch(function(err) {
            console.log('Ошибка сохранения в датасет:', err);
        });
    }

    function showRecommendedMove(uciMove) {
        recommendedMove = uciMove;
        render(false);
        
        // Добавляем подсказку в чат
        if (uciMove && uciMove.length === 4) {
            var from = uciMove.substring(0, 2);
            var to = uciMove.substring(2, 4);
            addChatMessage('💡 Рекомендую: <b>' + from + ' → ' + to + '</b>. Кликните на клетку ' + from + ' и сделайте ход.');
        }
    }

    function clearRecommendedMove() {
        recommendedMove = null;
        render(false);
    }

    function render(animate) {
        if (animate === undefined) animate = true;

        var legalDests = [];
        var legalMap = {};
        if (selected) {
            var s = rc(selected);
            legalDests = legalMoves(s.r, s.c);
            for (var i = 0; i < legalDests.length; i++) {
                legalMap[legalDests[i].r + ',' + legalDests[i].c] = legalDests[i];
            }
        }

        var checkSq = null;
        if (inCheck(turn)) {
            var k = findKing(turn);
            if (k) checkSq = k.r + ',' + k.c;
        }

        var topBar = document.querySelector('.top-bar');
        var topH = topBar ? topBar.offsetHeight : 58;
        var availH = window.innerHeight - topH;
        var availW = window.innerWidth * 0.6;
        var size = Math.floor(Math.min(availW - 60, availH - 160) / 8) * 8;
        boardEl.style.width = size + 'px';
        boardEl.style.height = size + 'px';

        boardEl.innerHTML = '';

        for (var r = 0; r < 8; r++) {
            for (var c = 0; c < 8; c++) {
                var sq = document.createElement('div');
                var isLight = (r + c) % 2 === 0;
                var name = sqName(r, c);
                sq.className = 'sq ' + (isLight ? 'light' : 'dark');
                sq.dataset.sq = name;

                if (name === selected) sq.classList.add('selected');
                if (name === lastFrom) sq.classList.add('last-from');
                if (name === lastTo) sq.classList.add('last-to');
                if (checkSq === r + ',' + c) sq.classList.add('check-sq');

                if (legalMap[r + ',' + c]) {
                    sq.classList.add(at(r, c) ? 'can-capture' : 'can-move');
                }

                sq.addEventListener('click', onSquareClick);
                boardEl.appendChild(sq);
            }
        }

        if (recommendedMove && recommendedMove.length === 4) {
            var recFrom = recommendedMove.substring(0, 2);
            var recTo = recommendedMove.substring(2, 4);
            var fromCoords = rc(recFrom);
            var toCoords = rc(recTo);
            var sqSize = size / 8;
            var arrow = document.createElement('div');
            arrow.className = 'ai-arrow';
            arrow.style.left = (fromCoords.c * sqSize + sqSize/2) + 'px';
            arrow.style.top = (fromCoords.r * sqSize + sqSize/2) + 'px';
            arrow.style.width = '20%';
            arrow.style.height = '20%';
            arrow.style.transform = 'translate(-50%, -50%)';
            boardEl.appendChild(arrow);
        }

        var animateFrom = null;
        var animateTo = null;
        var animatePiece = null;

        if (animate && lastMove) {
            animateFrom = { r: lastMove.fromR, c: lastMove.fromC };
            animateTo = { r: lastMove.toR, c: lastMove.toC };
            animatePiece = lastMove.piece;
        }

        for (var r = 0; r < 8; r++) {
            for (var c = 0; c < 8; c++) {
                if (animate && animateTo && r === animateTo.r && c === animateTo.c) continue;

                var piece = position[r][c];
                if (!piece) continue;

                if (animate && animateFrom && r === animateFrom.r && c === animateFrom.c) continue;

                createPieceElement(piece, r, c, size);
            }
        }

        if (animate && animateFrom && animateTo && animatePiece) {
            var movingPieceEl = createPieceElement(animatePiece, animateFrom.r, animateFrom.c, size);
            movingPieceEl.style.transition = 'none';
            
            setTimeout(function() {
                movingPieceEl.style.transition = 'left 0.2s ease-out, top 0.2s ease-out';
                movingPieceEl.style.left = (animateTo.c * 12.5) + '%';
                movingPieceEl.style.top = (animateTo.r * 12.5) + '%';
            }, 10);

            setTimeout(function() {
                lastMove = null;
            }, 250);
        } else {
            lastMove = null;
        }
    }

    function createPieceElement(piece, r, c, size) {
        var pieceEl = document.createElement('div');
        pieceEl.className = 'piece';

        var isWhite = colorOf(piece) === 'w';
        var fs = (size / 8 * 0.80);
        pieceEl.style.fontSize = fs + 'px';
        pieceEl.style.lineHeight = '1';
        pieceEl.style.pointerEvents = 'none';
        if (isWhite) {
            pieceEl.style.color = '#FFFFFF';
            pieceEl.style.textShadow =
                '-1px -1px 0 #1a1a1a, 1px -1px 0 #1a1a1a, ' +
                '-1px 1px 0 #1a1a1a, 1px 1px 0 #1a1a1a, ' +
                '0 0 6px rgba(0,0,0,0.5)';
        } else {
            pieceEl.style.color = '#1a1a1a';
            pieceEl.style.textShadow =
                '-1px -1px 0 #ccc, 1px -1px 0 #ccc, ' +
                '-1px 1px 0 #ccc, 1px 1px 0 #ccc, ' +
                '0 0 6px rgba(0,0,0,0.3)';
        }
        pieceEl.textContent = UNICODE[piece];

        pieceEl.style.left = (c * 12.5) + '%';
        pieceEl.style.top = (r * 12.5) + '%';

        boardEl.appendChild(pieceEl);
        return pieceEl;
    }

    function onSquareClick(e) {
        var name = e.currentTarget.dataset.sq;
        var pos = rc(name);

        if (selected) {
            var from = rc(selected);
            var legal = legalMoves(from.r, from.c);
            var move = null;
            for (var i = 0; i < legal.length; i++) {
                if (legal[i].r === pos.r && legal[i].c === pos.c) { move = legal[i]; break; }
            }

            if (move) {
                makeMove(from.r, from.c, move.r, move.c, move.special);
                selected = null;
                render(true);
                updateMoveList();
                updateStatus();

                if (!isGameOver() && turn === 'b') {
                    setTimeout(maiaMove, 500);
                }
                return;
            }

            var clickedPiece = at(pos.r, pos.c);
            if (clickedPiece && colorOf(clickedPiece) === turn) {
                selected = name;
                render(false);
                return;
            }

            selected = null;
            render(false);
            return;
        }

        var p = at(pos.r, pos.c);
        if (p && colorOf(p) === turn && turn === 'w') {
            selected = name;
            render(false);
        }
    }

    function isInsufficientMaterial() {
        var dominated = { w: [], b: [] };
        for (var r = 0; r < 8; r++) {
            for (var c = 0; c < 8; c++) {
                var p = at(r, c);
                if (!p) continue;
                dominated[colorOf(p)].push(typeOf(p));
            }
        }
        var w = dominated.w.sort().join('');
        var b = dominated.b.sort().join('');
        if (w === 'K' && b === 'K') return true;
        if (w === 'BK' && b === 'K') return true;
        if (w === 'K' && b === 'BK') return true;
        if (w === 'KN' && b === 'K') return true;
        if (w === 'K' && b === 'KN') return true;
        if (w === 'BK' && b === 'BK') return true;
        return false;
    }

    var halfmoveClock = 0;
    
// === НАВИГАЦИЯ ПО ХОДАМ ===

    var positionSnapshots = [];
    var viewIndex = -1;
    var isViewMode = false;

    function takeSnapshot() {
        var snap = {
            position: position.map(function(row) { return row.slice(); }),
            turn: turn,
            castling: { wK: castling.wK, wQ: castling.wQ, bK: castling.bK, bQ: castling.bQ },
            enPassant: enPassant,
            halfmoveClock: halfmoveClock,
            moveNumber: moveNumber,
            lastFrom: lastFrom,
            lastTo: lastTo
        };
        positionSnapshots.push(snap);
    }

    function restoreSnapshot(idx) {
        var snap = positionSnapshots[idx];
        position = snap.position.map(function(row) { return row.slice(); });
        turn = snap.turn;
        castling = { wK: snap.castling.wK, wQ: snap.castling.wQ, bK: snap.castling.bK, bQ: snap.castling.bQ };
        enPassant = snap.enPassant;
        halfmoveClock = snap.halfmoveClock;
        moveNumber = snap.moveNumber;
        lastFrom = snap.lastFrom;
        lastTo = snap.lastTo;
        selected = null;
        lastMove = null;
    }

    function navFirst() {
        if (positionSnapshots.length === 0) return;
        isViewMode = true; viewIndex = 0;
        restoreSnapshot(0);
        render(false);
        updateNavButtons();
        updateMoveListHighlight();
    }

    function navPrev() {
        if (positionSnapshots.length === 0) return;
        if (!isViewMode) { isViewMode = true; viewIndex = positionSnapshots.length - 1; }
        if (viewIndex > 0) viewIndex--;
        restoreSnapshot(viewIndex);
        render(false);
        updateNavButtons();
        updateMoveListHighlight();
    }

    function navNext() {
        if (!isViewMode) return;
        if (viewIndex < positionSnapshots.length - 1) {
            viewIndex++;
            restoreSnapshot(viewIndex);
            render(false);
        } else {
            isViewMode = false; viewIndex = -1;
            restoreSnapshot(positionSnapshots.length - 1);
            render(false);
        }
        updateNavButtons();
        updateMoveListHighlight();
    }

    function navLast() {
        if (positionSnapshots.length === 0) return;
        isViewMode = false; viewIndex = -1;
        restoreSnapshot(positionSnapshots.length - 1);
        render(false);
        updateNavButtons();
        updateMoveListHighlight();
    }

    function updateNavButtons() {
        var atStart = (isViewMode && viewIndex === 0) || positionSnapshots.length === 0;
        var atEnd = !isViewMode || viewIndex === positionSnapshots.length - 1;
        var first = document.getElementById('nav-first');
        var prev  = document.getElementById('nav-prev');
        var next  = document.getElementById('nav-next');
        var last  = document.getElementById('nav-last');
        if (!first) return;
        first.disabled = atStart; first.style.opacity = atStart ? '0.35' : '1';
        prev.disabled  = atStart; prev.style.opacity  = atStart ? '0.35' : '1';
        next.disabled  = atEnd;   next.style.opacity  = atEnd   ? '0.35' : '1';
        last.disabled  = atEnd;   last.style.opacity  = atEnd   ? '0.35' : '1';
    }

    function updateMoveListHighlight() {
        var spans = document.querySelectorAll('#move-list .move-pair span:not(.num)');
        spans.forEach(function(s) { s.style.background = ''; s.style.color = ''; });
        if (!isViewMode || positionSnapshots.length === 0) return;
        var moveIdx = viewIndex - 1;
        if (moveIdx < 0) return;
        var pairIdx = Math.floor(moveIdx / 2);
        var isWhiteMove = moveIdx % 2 === 0;
        var pairs = document.querySelectorAll('#move-list .move-pair');
        if (pairIdx >= pairs.length) return;
        var movespans = pairs[pairIdx].querySelectorAll('span:not(.num)');
        var target = movespans[isWhiteMove ? 0 : 1];
        if (target) { target.style.background = '#225A73'; target.style.color = '#fff'; }
    }

    document.addEventListener('keydown', function(e) {
        if (e.target.tagName === 'INPUT') return;
        if (e.key === 'ArrowLeft')  { e.preventDefault(); navPrev(); }
        if (e.key === 'ArrowRight') { e.preventDefault(); navNext(); }
    });
    // === КОНЕЦ НАВИГАЦИИ ===

    function isGameOver() {
        if (!hasAnyLegalMove(turn)) return true;
        if (isInsufficientMaterial()) return true;
        if (halfmoveClock >= 100) return true;
        return false;
    }

    function maiaMove() {
        var fen = toFEN();
        var elo = parseInt(document.getElementById('elo-slider').value);

        fetch('/api/stockfish-move', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({fen: fen, elo: elo})
        })
        .then(function(r) { return r.json(); })
        .then(function(data) {
            if (data.error) return;
            var from = rc(data.from);
            var to = rc(data.to);
            var special = null;
            var piece = at(from.r, from.c);
            if (piece && piece[1] === 'K' && Math.abs(from.c - to.c) === 2) {
                special = to.c > from.c ? 'castle-k' : 'castle-q';
            }
            if (piece && piece[1] === 'P' && from.c !== to.c && !at(to.r, to.c)) {
                special = 'en-passant';
            }
            if (piece && piece[1] === 'P' && (to.r === 0 || to.r === 7)) {
                special = 'promote';
            }
            makeMove(from.r, from.c, to.r, to.c, special);
            selected = null;
            render(true);
            updateMoveList();
            updateStatus();
        })
        .catch(function(err) {
            console.error('Maia2 ошибка:', err);
            var allMoves = [];
            for (var r = 0; r < 8; r++) {
                for (var c = 0; c < 8; c++) {
                    if (at(r, c) && colorOf(at(r, c)) === 'b') {
                        var mvs = legalMoves(r, c);
                        for (var i = 0; i < mvs.length; i++) {
                            allMoves.push({ fr: r, fc: c, tr: mvs[i].r, tc: mvs[i].c, special: mvs[i].special });
                        }
                    }
                }
            }
            if (allMoves.length === 0) return;
            var pick = allMoves[Math.floor(Math.random() * allMoves.length)];
            makeMove(pick.fr, pick.fc, pick.tr, pick.tc, pick.special);
            selected = null;
            render(true);
            updateMoveList();
            updateStatus();
        });
    }

    function updateFenDisplay() {
        document.getElementById('fen-input').value = toFEN();
    }

    function saveGame() {
        var state = {
            fen: toFEN(),
            moveHistory: moveHistory,
            moveNumber: moveNumber,
            capturedByWhite: capturedByWhite,
            capturedByBlack: capturedByBlack
        };
        localStorage.setItem('sfeducastling_game', JSON.stringify(state));
    }

    function loadSavedGame() {
        var saved = localStorage.getItem('sfeducastling_game');
        if (!saved) return false;
        try {
            var state = JSON.parse(saved);
            if (!state.fen) return false;
            loadFen(state.fen);
            if (state.capturedByWhite) capturedByWhite = state.capturedByWhite;
            if (state.capturedByBlack) capturedByBlack = state.capturedByBlack;
            updateMaterialDisplay();
            if (state.moveHistory) {
                moveHistory = state.moveHistory;
                moveNumber = state.moveNumber || 1;
                updateMoveList();
            }
            return true;
        } catch(e) { return false; }
    }

    function updateMoveList() {
        var container = document.getElementById('move-list');
        container.innerHTML = '';
        for (var i = 0; i < moveHistory.length; i++) {
            var pair = document.createElement('span');
            pair.className = 'move-pair';
            var html = '<span class="num">' + moveHistory[i].num + '.</span> '
                + '<span>' + moveHistory[i].w + '</span>';
            if (moveHistory[i].b) {
                html += ' <span>' + moveHistory[i].b + '</span>';
            }
            pair.innerHTML = html;
            container.appendChild(pair);
        }
        container.scrollTop = container.scrollHeight;
    }

    function updateStatus() {
        updateFenDisplay();
        updateMaterialDisplay();
        saveGame();
        var el = document.getElementById('game-status');
        var noMoves = !hasAnyLegalMove(turn);
        if (noMoves && inCheck(turn)) {
            el.textContent = turn === 'w' ? 'Мат — чёрные победили' : 'Мат — белые победили!';
            addChatMessage('Партия окончена. Отличная игра!');
        } else if (noMoves) {
            el.textContent = 'Пат — ничья';
            addChatMessage('Партия завершилась вничью (пат).');
        } else if (isInsufficientMaterial()) {
            el.textContent = 'Ничья — недостаточно материала';
            addChatMessage('Ничья: недостаточно фигур для мата.');
        } else if (halfmoveClock >= 100) {
            el.textContent = 'Ничья — правило 50 ходов';
            addChatMessage('Ничья по правилу 50 ходов без взятий и ходов пешками.');
        } else if (inCheck(turn)) {
            el.textContent = turn === 'w' ? 'Шах! Ваш ход' : 'Шах! Maia думает...';
        } else {
            el.textContent = turn === 'w' ? 'Ваш ход' : 'Maia думает...';
        }
    }

    function formatMarkdown(text) {
        var d = document.createElement('div');
        d.textContent = text;
        var s = d.innerHTML;
        s = s.replace(/^### (.+)$/gm, '<div style="font-weight:700;color:#83b4aa;margin-top:8px;">$1</div>');
        s = s.replace(/^## (.+)$/gm, '<div style="font-weight:700;color:#83b4aa;font-size:15px;margin-top:8px;">$1</div>');
        s = s.replace(/^# (.+)$/gm, '<div style="font-weight:700;color:#83b4aa;font-size:16px;margin-top:8px;">$1</div>');
        s = s.replace(/\*\*(.+?)\*\*/g, '<b style="color:#e2e8f0;">$1</b>');
        s = s.replace(/\*(.+?)\*/g, '<i>$1</i>');
        s = s.replace(/^[-•] (.+)$/gm, '<div style="padding-left:12px;">• $1</div>');
        s = s.replace(/\n/g, '<br>');
        return s;
    }

    function addChatMessage(text) {
        var container = document.getElementById('chat-messages');
        var msg = document.createElement('div');
        msg.className = 'chat-msg';
        msg.innerHTML = '<div class="msg-avatar">⚡</div><div class="msg-body">' + formatMarkdown(text) + '</div>';
        container.appendChild(msg);
        container.scrollTop = container.scrollHeight;
    }

    function escapeHtml(str) {
        var d = document.createElement('div');
        d.textContent = str;
        return d.innerHTML;
    }

    // === Парсинг ходов из текста ===
    function parseChessMove(text) {
        // Ищем ходы в UCI формате (e2e4, d7d5, g1f3)
        var uciMatch = text.match(/([a-h][1-8])([a-h][1-8])/i);
        if (uciMatch) {
            return uciMatch[1].toLowerCase() + uciMatch[2].toLowerCase();
        }
        
        // Ищем ходы пешки (e4, d5)
        var sanMatch = text.match(/([a-h][2-7])\s*[-→x]?\s*([a-h][1-8])/i);
        if (sanMatch) {
            return sanMatch[1].toLowerCase() + sanMatch[2].toLowerCase();
        }
        
        // Ищем фигурные ходы (Nf3, Bxc6, O-O, O-O-O)
        var pieceMatch = text.match(/([KQRBN])[a-h]?[1-8]?\s*[-→x]?\s*([a-h][1-8])/i);
        if (pieceMatch) {
            // Для точного преобразования нужно знать позицию, 
            // но показываем хотя бы клетку назначения
            return null;
        }
        
        return null;
    }

    // === Отправка сообщений в чат (GigaChat) ===
    function sendChat() {
        var input = document.getElementById('chat-input');
        var text = input.value.trim();
        if (!text) return;
        input.value = '';
        var container = document.getElementById('chat-messages');
        var userMsg = document.createElement('div');
        userMsg.className = 'chat-msg';
        userMsg.innerHTML = '<div class="msg-avatar" style="background:#0d3550">👤</div>'
            + '<div class="msg-body" style="border-radius:12px 0 12px 12px">' + escapeHtml(text) + '</div>';
        container.appendChild(userMsg);
        container.scrollTop = container.scrollHeight;
        
        fetch('/api/analyze', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({fen: toFEN()})
        })
        .then(function(r) { return r.json(); })
        .then(function(data) {
            var response = data.message || 'Нет ответа от GigaChat.';
            addChatMessage(response);
            
            // Пробуем найти ход в ответе и показать на доске
            var move = parseChessMove(response);
            if (move) {
                showRecommendedMove(move);
            }
        })
        .catch(function() {
            addChatMessage('Ошибка соединения с сервером.');
        });
    }

    document.getElementById('chat-send').addEventListener('click', sendChat);
    document.getElementById('chat-input').addEventListener('keydown', function(e) {
        if (e.key === 'Enter') sendChat();
    });

    // === Кнопка анализа позиции ===
    document.getElementById('analyze-btn').addEventListener('click', function() {
        var fen = toFEN();
        var container = document.getElementById('chat-messages');
        
        // Добавляем сообщение о начале анализа
        var msg = document.createElement('div');
        msg.className = 'chat-msg';
        msg.innerHTML = '<div class="msg-avatar">🔍</div>'
            + '<div class="msg-body">Анализ позиции через Stockfish...</div>';
        container.appendChild(msg);
        container.scrollTop = container.scrollHeight;
        
        // Отправляем запрос на анализ
        fetch('/api/stockfish-analyze', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({fen: fen, elo: 1500})
        })
        .then(function(r) { 
            if (!r.ok) {
                throw new Error('HTTP ' + r.status);
            }
            return r.json(); 
        })
        .then(function(data) {
            console.log('Ответ от API:', data); // Отладка
            
            if (data.error) {
                addChatMessage('Ошибка: ' + data.error);
            } else {
                var evalText = '';
                if (data.evaluation && data.evaluation.type === 'cp') {
                    evalText = data.evaluation.value > 0 
                        ? '+' + (data.evaluation.value / 100).toFixed(1) 
                        : (data.evaluation.value / 100).toFixed(1);
                } else if (data.evaluation && data.evaluation.type === 'mate') {
                    evalText = 'Мат в ' + data.evaluation.value;
                } else {
                    evalText = 'без оценки';
                }
                
                // Форматируем варианты (учитываем разный регистр поля в API)
                var variants = '';
                if (data.top_moves && data.top_moves.length > 0) {
                    variants = data.top_moves.slice(0, 3).map(function(m) {
                        return m.Move || m.move;
                    }).join(', ');
                }
                
                var msg = document.createElement('div');
                msg.className = 'chat-msg';
                msg.innerHTML = '<div class="msg-avatar">🔍</div>'
                    + '<div class="msg-body">'
                    + '<b>Анализ Stockfish:</b><br>'
                    + '• Лучший ход: <b>' + data.best_move + '</b><br>'
                    + '• Оценка: <b>' + evalText + '</b><br>'
                    + '• Варианты: ' + variants
                    + '</div>';
                container.appendChild(msg);
                container.scrollTop = container.scrollHeight;
                
                // Показываем лучший ход на доске
                if (data.best_move) {
                    showRecommendedMove(data.best_move.toLowerCase());
                }
            }
        })
        .catch(function(err) {
            console.error('Ошибка анализа:', err);
            addChatMessage('Ошибка: ' + err.message);
        });
    });

    function loadFen(fen) {
        fen = fen.trim();
        if (!fen) return;
        var parts = fen.split(' ');
        var rows = parts[0].split('/');
        if (rows.length !== 8) return;

        position = [];
        var pieceMap = {
            'r':'bR','n':'bN','b':'bB','q':'bQ','k':'bK','p':'bP',
            'R':'wR','N':'wN','B':'wB','Q':'wQ','K':'wK','P':'wP'
        };
        for (var r = 0; r < 8; r++) {
            position[r] = [];
            var col = 0;
            for (var i = 0; i < rows[r].length; i++) {
                var ch = rows[r][i];
                if (ch >= '1' && ch <= '8') {
                    var empty = parseInt(ch);
                    for (var e = 0; e < empty; e++) {
                        position[r][col] = null;
                        col++;
                    }
                } else if (pieceMap[ch]) {
                    position[r][col] = pieceMap[ch];
                    col++;
                }
            }
        }

        turn = (parts.length > 1 && parts[1] === 'b') ? 'b' : 'w';
        selected = null;
        lastFrom = null;
        lastTo = null;
        lastMove = null;
        moveHistory = [];
        moveNumber = 1;
        capturedByWhite = [];
        capturedByBlack = [];
        castling = { wK: false, wQ: false, bK: false, bQ: false };
        if (parts.length > 2) {
            var c = parts[2];
            if (c.indexOf('K') !== -1) castling.wK = true;
            if (c.indexOf('Q') !== -1) castling.wQ = true;
            if (c.indexOf('k') !== -1) castling.bK = true;
            if (c.indexOf('q') !== -1) castling.bQ = true;
        }
        enPassant = (parts.length > 3 && parts[3] !== '-') ? parts[3] : null;
        halfmoveClock = (parts.length > 4) ? parseInt(parts[4]) || 0 : 0;

        document.getElementById('fen-input').value = fen;
        document.getElementById('move-list').innerHTML = '';
        render(false);
        updateStatus();
        updateMaterialDisplay();
        addChatMessage('Позиция загружена из FEN.');
    }

    document.getElementById('fen-load').addEventListener('click', function() {
        loadFen(document.getElementById('fen-input').value);
    });
    document.getElementById('fen-input').addEventListener('keydown', function(e) {
        if (e.key === 'Enter') loadFen(e.target.value);
    });
    document.getElementById('fen-file').addEventListener('click', function() {
        document.getElementById('fen-file-input').click();
    });
    document.getElementById('fen-file-input').addEventListener('change', function(e) {
        var file = e.target.files[0];
        if (!file) return;
        var reader = new FileReader();
        reader.onload = function(ev) {
            var line = ev.target.result.trim().split('\n')[0].trim();
            if (line) loadFen(line);
        };
        reader.readAsText(file);
        e.target.value = '';
    });

    window.addEventListener('resize', function() { render(false); });

    document.getElementById('new-game-btn').addEventListener('click', function() {
        localStorage.removeItem('sfeducastling_game');
        init();
        document.getElementById('chat-messages').innerHTML = '';
        addChatMessage('Новая игра начата. Удачи!');
    });

    document.getElementById('flip-board-btn').addEventListener('click', function() {
        boardEl.classList.toggle('flipped');
        boardEl.parentElement.classList.toggle('flipped');
    });

    document.getElementById('elo-slider').addEventListener('input', function() {
        document.getElementById('elo-display').textContent = this.value;
    });

    // === Кнопка изучения дебюта ===
    document.getElementById('learn-btn').addEventListener('click', function() {
        var container = document.getElementById('chat-messages');
        
        // Показываем сообщение о начале
        addChatMessage('📚 Хотите изучить дебют? Загружаю случайную позицию из базы...');
        
        // Запрашиваем случайный дебют
        fetch('/api/knowledge/random-opening')
        .then(function(r) { return r.json(); })
        .then(function(data) {
            if (data.error) {
                addChatMessage('Ошибка: ' + data.error);
                return;
            }
            
            var opening = data.opening;
            if (opening) {
                var msg = '📚 <b>' + opening.name + '</b> (' + (opening.eco || 'ECO') + ')\n\n';
                msg += opening.description + '\n\n';
                msg += '<b>Идеи:</b>\n';
                if (opening.ideas) {
                    opening.ideas.forEach(function(idea) {
                        msg += '• ' + idea + '\n';
                    });
                }
                addChatMessage(msg);
                
                // Загружаем позицию на доску
                if (opening.fen) {
                    loadFen(opening.fen);
                    addChatMessage('Позиция загружена. Попробуйте сделать ходы по теории!');
                }
            }
        })
        .catch(function(err) {
            addChatMessage('Ошибка загрузки базы знаний: ' + err.message);
        });
    });

    // Очищаем рекомендацию при клике на доску
    boardEl.addEventListener('click', function() {
        clearRecommendedMove();
    });

    // Экспортируем функции глобально для index.html
    window.maiaMove = maiaMove;
    window.loadFen = loadFen;
    window.addChatMessage = addChatMessage;
    window.init = init;
    window.toFEN = toFEN;
    window.navFirst = navFirst;
    window.navPrev  = navPrev;
    window.navNext  = navNext;
    window.navLast  = navLast;

    if (!loadSavedGame()) {
        init();
    }
})();
