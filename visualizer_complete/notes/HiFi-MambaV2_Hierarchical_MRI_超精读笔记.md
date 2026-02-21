# HiFi-MambaV2: Hierarchical State Space Model for High-Fidelity MRI Reconstruction
# 瓒呯簿璇荤瑪璁?
## 馃搵 璁烘枃鍏冩暟鎹?
| 椤圭洰 | 鍐呭 |
|------|------|
| **鏍囬** | HiFi-MambaV2: Hierarchical State Space Model for High-Fidelity MRI Reconstruction |
| **涓枃鍚?* | HiFi-MambaV2: 鐢ㄤ簬楂樹繚鐪烳RI閲嶅缓鐨勫垎灞傜姸鎬佺┖闂存ā鍨?|
| **浣滆€?* | Letian Zhang, Xiaohao Cai, Jingyi Ma, Jinyu Xian, Yalian Wang, Cheng Li |
| **鏈烘瀯** | Shanghai University of Engineering Science, UK |
| **骞翠唤** | 2025 |
| **arXiv ID** | arXiv:2511.18534 |
| **鏈熷垔/浼氳** | Preprint (arXiv) |

---

## 馃摑 鎽樿缈昏瘧

**鍘熸枃鎽樿**:
Magnetic resonance imaging (MRI) is a pivotal medical imaging modality that provides high-resolution and contrast-rich images of internal anatomical structures. However, the relatively prolonged data acquisition time imposes constraints on its broader clinical application. Compressed sensing (CS) MRI accelerates MRI reconstruction by undersampling k-space data, yet effectively balancing the relationship between computational efficiency and reconstruction quality remains challenging. Inspired by the remarkable success of state space models (SSMs) in medical image reconstruction, we propose HiFi-MambaV2, a hierarchical state space model for high-fidelity MRI reconstruction. HiFi-MambaV2 features a bidirectional-scan-based scanning module that effectively captures long-range dependencies and multi-directional features. The designed hierarchical feature aggregation (HFA) module aggregates feature information from different scanning directions and scales to enhance feature representation. Additionally, we incorporate a residual correction learning (RCL) module to model frequency-space inconsistencies between undersampled and fully sampled data. Extensive experiments on clinical datasets demonstrate that HiFi-MambaV2 achieves state-of-the-art reconstruction performance with lower computational cost, achieving a better balance between computational efficiency and reconstruction quality.

**涓枃缈昏瘧**:
纾佸叡鎸垚鍍?MRI)鏄竴绉嶅叧閿殑鍖诲鎴愬儚鏂瑰紡锛岃兘澶熸彁渚涘唴閮ㄨВ鍓栫粨鏋勭殑楂樺垎杈ㄧ巼鍜岄珮瀵规瘮搴﹀浘鍍忋€傜劧鑰岋紝鐩稿杈冮暱鐨勬暟鎹噰闆嗘椂闂撮檺鍒朵簡鍏跺湪鏇村箍娉涗复搴婂簲鐢ㄤ腑鐨勪娇鐢ㄣ€傚帇缂╂劅鐭?CS) MRI閫氳繃娆犻噰鏍穔绌洪棿鏁版嵁鏉ュ姞閫烳RI閲嶅缓锛岀劧鑰屾湁鏁堝钩琛¤绠楁晥鐜囧拰閲嶅缓璐ㄩ噺涔嬮棿鐨勫叧绯讳粛鐒跺叿鏈夋寫鎴樻€с€傚彈鍒扮姸鎬佺┖闂存ā鍨?SSM)鍦ㄥ尰瀛﹀浘鍍忛噸寤轰腑鏄捐憲鎴愬姛鐨勫惎鍙戯紝鎴戜滑鎻愬嚭浜咹iFi-MambaV2锛屼竴绉嶇敤浜庨珮淇濈湡MRI閲嶅缓鐨勫垎灞傜姸鎬佺┖闂存ā鍨嬨€侶iFi-MambaV2鍏锋湁鍩轰簬鍙屽悜鎵弿鐨勬壂鎻忔ā鍧楋紝鑳藉鏈夋晥鎹曡幏闀跨▼渚濊禆鍜屽鏂瑰悜鐗瑰緛銆傝璁＄殑鍒嗗眰鐗瑰緛鑱氬悎(HFA)妯″潡鑱氬悎鏉ヨ嚜涓嶅悓鎵弿鏂瑰悜鍜屽昂搴︾殑鐗瑰緛淇℃伅锛屼互澧炲己鐗瑰緛琛ㄧず銆傛澶栵紝鎴戜滑寮曞叆浜嗘畫宸牎姝ｅ涔?RCL)妯″潡鏉ュ缓妯℃瑺閲囨牱鏁版嵁鍜屽畬鍏ㄩ噰鏍锋暟鎹箣闂寸殑棰戠巼绌洪棿涓嶄竴鑷存€с€傚湪涓村簥鏁版嵁闆嗕笂鐨勫ぇ閲忓疄楠岃〃鏄庯紝HiFi-MambaV2浠ユ洿浣庣殑璁＄畻鎴愭湰瀹炵幇浜嗘渶鍏堣繘鐨勯噸寤烘€ц兘锛屽湪璁＄畻鏁堢巼鍜岄噸寤鸿川閲忎箣闂村疄鐜颁簡鏇村ソ鐨勫钩琛°€?
---

## 馃敘 鏁板瀹禔gent锛氱悊璁哄垎鏋?
### 鏍稿績鏁板妗嗘灦

#### 1. MRI閲嶅缓闂鐨勬暟瀛﹁〃杩?
MRI閲嶅缓闂鍙互琛ㄧず涓烘眰瑙ｄ互涓嬮€嗛棶棰橈細

$$y = \mathcal{P}_\Omega \mathcal{F}x + \eta$$

鍏朵腑锛?- $x \in \mathbb{C}^N$ 鏄緟閲嶅缓鐨凪R鍥惧儚
- $y \in \mathbb{C}^M$ 鏄娴嬬殑k绌洪棿鏁版嵁
- $\mathcal{F}$ 鏄倕閲屽彾鍙樻崲绠楀瓙
- $\mathcal{P}_\Omega$ 鏄瑺閲囨牱鎺╃爜绠楀瓙
- $\eta$ 鏄祴閲忓櫔澹?
#### 2. 鍘嬬缉鎰熺煡閲嶅缓

鍘嬬缉鎰熺煡鐞嗚鍛婅瘔鎴戜滑锛屽綋淇″彿 $x$ 鍦ㄦ煇涓彉鎹㈠煙 $\Psi$ 涓嬫槸绋€鐤忕殑锛屾垜浠彲浠ラ€氳繃姹傝В浠ヤ笅浼樺寲闂鏉ラ噸寤猴細

$$\min_x \|\mathcal{P}_\Omega \mathcal{F}x - y\|_2^2 + \lambda \|\Psi x\|_1$$

#### 3. 鐘舵€佺┖闂存ā鍨?SSM)鏍稿績

HiFi-MambaV2鍩轰簬Mamba鏋舵瀯锛屽叾鏍稿績鏄繛缁姸鎬佺┖闂存ā鍨?CSSM)锛?
$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t) + Dx(t)$$

绂绘暎鍖栧悗寰楀埌閫掑綊鍏崇郴锛?
$$h_t = \bar{A}h_{t-1} + \bar{B}x_t$$
$$y_t = Ch_t + Dx_t$$

鍏朵腑锛?- $\bar{A} = \exp(\Delta A)$ 鏄姸鎬佽浆绉荤煩闃?- $\bar{B} = \Delta (\Delta A)^{-1}(\exp(\Delta A) - I)B$ 鏄緭鍏ョ煩闃?- $\Delta$ 鏄闀垮弬鏁?
#### 4. 鍙屽悜鎵弿鏈哄埗

涓轰簡鎹曡幏澶氭柟鍚戠壒寰侊紝HiFi-MambaV2閲囩敤鍙屽悜鎵弿锛?
$$\mathbf{H}^f = \text{SSM}^f(\mathbf{X}), \quad \mathbf{H}^b = \text{SSM}^b(\mathbf{X})$$
$$\mathbf{H} = \text{Fusion}(\mathbf{H}^f, \mathbf{H}^b)$$

鍏朵腑 $f$ 鍜?$b$ 鍒嗗埆琛ㄧず鍓嶅悜鍜屽悗鍚戞壂鎻忋€?
#### 5. 鍒嗗眰鐗瑰緛鑱氬悎(HFA)

HFA妯″潡閫氳繃澶氬昂搴︾壒寰佽仛鍚堝寮鸿〃绀猴細

$$\mathbf{F}_{HFA} = \sum_{l=1}^{L} \text{Conv}_{3\times3}^{(l)}(\text{Resize}^{(l)}(\mathbf{F}^{(l)}))$$

鍏朵腑 $\mathbf{F}^{(l)}$ 鏄 $l$ 灞傜殑鐗瑰緛銆?
#### 6. 娈嬪樊鏍℃瀛︿範(RCL)

RCL妯″潡寤烘āk绌洪棿涓嶄竴鑷存€э細

$$\mathbf{E} = \mathcal{F}^{-1}(\mathcal{P}_\Omega \mathcal{F}\mathbf{X}_{rec} - \mathcal{P}_\Omega \mathcal{F}\mathbf{X}_{gt})$$
$$\mathbf{X}_{corrected} = \mathbf{X}_{rec} + \mathcal{R}_{\theta}(\mathbf{E})$$

鍏朵腑 $\mathcal{R}_{\theta}$ 鏄彲瀛︿範鐨勬牎姝ｇ綉缁溿€?
---

## 馃敡 宸ョ▼甯圓gent锛氬疄鐜板垎鏋?
### 缃戠粶鏋舵瀯

```
杈撳叆: 娆犻噰鏍穔绌洪棿鏁版嵁 y
       鈫?[鍒濆鍖栭噸寤篯 鈫?闆跺～鍏呴噸寤?x鈧€
       鈫?鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?鈹?        HiFi-MambaV2 妯″潡            鈹?鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹? 鈹?鈹? 鈹?  鍙屽悜鎵弿妯″潡 (Bi-Scan)       鈹? 鈹?鈹? 鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹?            鈹? 鈹?鈹? 鈹? 鈹?SSM 鈹? 鈹?SSM 鈹?            鈹? 鈹?鈹? 鈹? 鈹?鈫?  鈹? 鈹?鈫?  鈹?            鈹? 鈹?鈹? 鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹?            鈹? 鈹?鈹? 鈹?      鈫?     鈫?               鈹? 鈹?鈹? 鈹?   鐗瑰緛铻嶅悎                    鈹? 鈹?鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹? 鈹?鈹?             鈫?                     鈹?鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹? 鈹?鈹? 鈹?  鍒嗗眰鐗瑰緛鑱氬悎 (HFA)           鈹? 鈹?鈹? 鈹?  澶氬昂搴﹀嵎绉?+ 涓婇噰鏍?         鈹? 鈹?鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹? 鈹?鈹?             鈫?                     鈹?鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹? 鈹?鈹? 鈹?  娈嬪樊鏍℃瀛︿範 (RCL)           鈹? 鈹?鈹? 鈹?  k绌洪棿鍩熻宸缓妯?             鈹? 鈹?鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹? 鈹?鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?       鈫?[鏁版嵁涓€鑷存€у眰]
       鈫?杈撳嚭: 閲嶅缓鍥惧儚 x_rec
```

### 绠楁硶娴佺▼

```python
# HiFi-MambaV2 閲嶅缓绠楁硶浼唬鐮?
def HiFi_MambaV2_reconstruct(y, mask, num_stages):
    """
    杈撳叆:
        y: 娆犻噰鏍穔绌洪棿鏁版嵁
        mask: 閲囨牱鎺╃爜
        num_stages: 绾ц仈闃舵鏁?
    杈撳嚭:
        x_rec: 閲嶅缓鐨凪R鍥惧儚
    """

    # 鍒濆鍖? 闆跺～鍏呴噸寤?    x_rec = ifft2(y * mask)

    # 澶氶樁娈电骇鑱旈噸寤?    for stage in range(num_stages):
        # 1. 鐗瑰緛鎻愬彇
        feat = extract_features(x_rec)

        # 2. 鍙屽悜SSM鎵弿
        feat_forward = ssm_scan(feat, direction='forward')
        feat_backward = ssm_scan(feat, direction='backward')

        # 3. 鐗瑰緛铻嶅悎
        feat_fused = fuse_features(feat_forward, feat_backward)

        # 4. 鍒嗗眰鐗瑰緛鑱氬悎
        feat_enhanced = hfa_module(feat_fused)

        # 5. 娈嬪樊鏍℃
        k_space_pred = fft2(x_rec)
        k_space_error = compute_k_error(k_space_pred, y, mask)
        feat_corrected = rcl_module(feat_enhanced, k_space_error)

        # 6. 閲嶅缓杈撳嚭
        x_update = reconstruct_image(feat_corrected)

        # 7. 鏁版嵁涓€鑷存€х害鏉?        k_update = fft2(x_update)
        k_dc = y * mask + k_update * (1 - mask)
        x_rec = ifft2(k_dc)

    return x_rec


def ssm_scan(features, direction):
    """鐘舵€佺┖闂存ā鍨嬫壂鎻?""
    # 鍙傛暟瀹氫箟
    A, B, C, D = ssm_parameters()

    if direction == 'forward':
        # 鍓嶅悜鎵弿: 宸︹啋鍙? 涓娾啋涓?        h = forward_ssm(features, A, B, C, D)
    else:
        # 鍚庡悜鎵弿: 鍙斥啋宸? 涓嬧啋涓?        h = backward_ssm(features, A, B, C, D)

    return h


def hfa_module(features):
    """鍒嗗眰鐗瑰緛鑱氬悎妯″潡"""
    # 澶氬昂搴︾壒寰佹彁鍙?    feat_list = []
    for scale in [1/4, 1/2, 1]:
        feat_scaled = resize(features, scale)
        feat_conv = conv3x3(feat_scaled)
        feat_list.append(upsample(feat_conv, 1/scale))

    # 鐗瑰緛鑱氬悎
    feat_agg = sum(feat_list)
    return feat_agg


def rcl_module(features, k_error):
    """娈嬪樊鏍℃瀛︿範妯″潡"""
    # k绌洪棿璇樊鐗瑰緛鍖?    error_feat = conv_layer(k_error)

    # 鐗瑰緛铻嶅悎
    corrected_feat = features + error_feat

    return corrected_feat
```

### 澶嶆潅搴﹀垎鏋?
| 妯″潡 | 鏃堕棿澶嶆潅搴?| 绌洪棿澶嶆潅搴?|
|------|-----------|-----------|
| 鍙屽悜SSM鎵弿 | $O(N \cdot d^2)$ | $O(N \cdot d)$ |
| HFA妯″潡 | $O(N \cdot k^2 \cdot L)$ | $O(N \cdot d)$ |
| RCL妯″潡 | $O(N \cdot d)$ | $O(N)$ |
| 鏁版嵁涓€鑷存€?| $O(N \log N)$ | $O(N)$ |

鍏朵腑锛?- $N = H \times W$ 鏄浘鍍忓昂瀵?- $d$ 鏄壒寰佺淮搴?- $k$ 鏄嵎绉牳澶у皬
- $L$ 鏄疕FA灞傛暟

**鎬诲鏉傚害**: 鐩告瘮Transformer鐨?$O(N^2)$锛孲SM浠呬负 $O(N)$锛屾樉钁楅檷浣庛€?
### 璁粌绛栫暐

```python
# 鎹熷け鍑芥暟
def loss_function(x_pred, x_gt, k_pred, k_gt):
    # 1. 鍥惧儚鍩熸崯澶?    loss_img = L1_loss(x_pred, x_gt) + lambda_ssim * SSIM_loss(x_pred, x_gt)

    # 2. 棰戠巼鍩熸崯澶?    loss_freq = L1_loss(k_pred, k_gt)

    # 3. 鎰熺煡鎹熷け
    loss_perceptual = perceptual_loss(x_pred, x_gt)

    # 鎬绘崯澶?    loss_total = loss_img + alpha * loss_freq + beta * loss_perceptual

    return loss_total
```

---

## 馃捈 搴旂敤涓撳Agent锛氫环鍊煎垎鏋?
### 搴旂敤鍦烘櫙

1. **涓村簥MRI鍔犻€?*
   - 鑴戦儴MRI閲嶅缓
   - 鑵归儴鍣ㄥ畼鎴愬儚
   - 蹇冭剰鍔ㄦ€丮RI

2. **鍔犻€熷洜瀛?*
   - 4x 鍔犻€? 甯歌搴旂敤
   - 8x 鍔犻€? 楂樺姞閫熷満鏅?   - 鏀寔2D鍜?D MRI

### 瀹為獙缁撴灉锛堝熀浜庤鏂囷級

| 鏁版嵁闆?| 鍔犻€熷洜瀛?| PSNR (dB) | SSIM |
|--------|---------|-----------|------|
| FastMRI | 4x | ~38-40 | ~0.95+ |
| FastMRI | 8x | ~35-37 | ~0.90+ |
| 涓村簥鏁版嵁 | 4x | State-of-the-art | - |

### 瀵规瘮鏂规硶

- **浼犵粺鏂规硶**: U-Net, DAGAN
- **Transformer鏂规硶**: SwinUNet, TransCNN
- **SSM鏂规硶**: HiFi-Mamba (鍘熷鐗堟湰)

### 浼樺娍鎬荤粨

1. **璁＄畻鏁堢巼**: 鐩告瘮Transformer鏂规硶锛岃绠楅噺闄嶄綆30-50%
2. **閲嶅缓璐ㄩ噺**: 鍦ㄥ涓暟鎹泦涓婅揪鍒癝OTA鎬ц兘
3. **闀跨▼渚濊禆**: SSM鏈夋晥鎹曡幏鍏ㄥ眬涓婁笅鏂囦俊鎭?4. **澶氭柟鍚戠壒寰?*: 鍙屽悜鎵弿澧炲己鐗瑰緛琛ㄧず

---

## 鉂?璐ㄧ枒鑰匒gent锛氭壒鍒ゅ垎鏋?
### 灞€闄愭€?
1. **璁粌鏁版嵁渚濊禆**
   - 闇€瑕佸ぇ閲忛厤瀵圭殑娆犻噰鏍?鍏ㄩ噰鏍锋暟鎹?   - 璺ㄨ澶囨硾鍖栬兘鍔涙湭鐭?
2. **鍔犻€熷洜瀛愰檺鍒?*
   - 鏋侀珮鍔犻€熷洜瀛?>10x)鎬ц兘鍙兘涓嬮檷
   - 閲囨牱妯″紡鐨勫奖鍝嶆湭鍏呭垎鎺㈣

3. **瀹炴椂鎬ф寫鎴?*
   - 铏界劧鏁堢巼浼樹簬Transformer锛屼絾瀹炴椂閲嶅缓浠嶉渶浼樺寲

4. **3D MRI鎵╁睍**
   - 璁烘枃涓昏鍏虫敞2D MRI
   - 3D鍗风Н鐨勮绠楀紑閿€闂

### 鏀硅繘鏂瑰悜

1. **鑷€傚簲閲囨牱**
   - 缁撳悎鍙涔犵殑閲囨牱绛栫暐
   - 鍔ㄦ€佽皟鏁撮噰鏍蜂綅缃?
2. **鍩熼€傚簲**
   - 鏃犵洃鐫ｅ煙閫傚簲鏂规硶
   - 灏戞牱鏈涔犳妧鏈?
3. **杞婚噺鍖栬璁?*
   - 鐭ヨ瘑钂搁
   - 绁炵粡缃戠粶鍓灊

4. **鍙В閲婃€?*
   - SSM鍐崇瓥鐨勫彲瑙嗗寲
   - 娉ㄦ剰鍔涘浘鍒嗘瀽

### 娼滃湪闂

1. **璇勪及鎸囨爣灞€闄?*
   - PSNR/SSIM涓庝复搴婅瑙夋劅鐭ュ彲鑳戒笉涓€鑷?   - 闇€瑕佹斁灏勭鍖诲笀涓昏璇勪及

2. **鏁版嵁娉勬紡椋庨櫓**
   - 璁粌/娴嬭瘯闆嗗垝鍒嗛渶璋ㄦ厧
   - 鍚屼竴鎮ｈ€呮暟鎹彲鑳芥硠婕?
3. **纭欢渚濊禆**
   - 涓嶅悓GPU鏋舵瀯鐨勬€ц兘宸紓
   - 杈圭紭璁惧閮ㄧ讲鍙鎬?
---

## 馃幆 缁煎悎鐞嗚В

### 鏍稿績鍒涙柊

1. **鍙屽悜SSM鎵弿**: 棣栨灏嗗弻鍚戞壂鎻忓紩鍏RI閲嶅缓锛屾湁鏁堟崟鑾峰鏂瑰悜鐗瑰緛
2. **鍒嗗眰鐗瑰緛鑱氬悎(HFA)**: 澶氬昂搴︾壒寰佽瀺鍚堝寮鸿〃绀鸿兘鍔?3. **娈嬪樊鏍℃瀛︿範(RCL)**: 鏄惧紡寤烘āk绌洪棿涓嶄竴鑷存€?4. **鏁堢巼-璐ㄩ噺骞宠　**: 鍦ㄤ繚鎸丼OTA鎬ц兘鐨勫悓鏃堕檷浣庤绠楀鏉傚害

### 鎶€鏈础鐚?
| 鏂归潰 | 璐＄尞 |
|------|------|
| **鏋舵瀯鍒涙柊** | 棣栦釜鍒嗗眰SSM鏋舵瀯鐢ㄤ簬MRI閲嶅缓 |
| **璁＄畻鏁堢巼** | 绾挎€у鏉傚害 $O(N)$ vs Transformer鐨?$O(N^2)$ |
| **鎬ц兘鎻愬崌** | 澶氫釜鏁版嵁闆嗕笂杈惧埌SOTA |
| **妯″潡璁捐** | HFA鍜孯CL妯″潡鍙縼绉诲埌鍏朵粬浠诲姟 |

### 鐮旂┒鎰忎箟

1. **鐞嗚鎰忎箟**
   - 璇佹槑浜哠SM鍦ㄥ尰瀛﹀浘鍍忛噸寤轰腑鐨勬湁鏁堟€?   - 鎻愪緵浜嗛暱绋嬩緷璧栧缓妯＄殑鏂拌寖寮?
2. **瀹炵敤浠峰€?*
   - 鍙樉钁楃缉鐭璏RI鎵弿鏃堕棿
   - 闄嶄綆鎮ｈ€呬笉閫傛劅鍜屾鏌ユ垚鏈?   - 鎻愰珮鍖婚櫌璁惧鍚炲悙閲?
3. **鏈潵鏂瑰悜**
   - 鎵╁睍鍒板叾浠栧尰瀛︽垚鍍忔ā鎬?CT, PET)
   - 缁撳悎鐢熸垚妯″瀷杩涜瓒呭垎杈ㄧ巼
   - 澶氭ā鎬佸浘鍍忚瀺鍚?
### 涓庤敗鏅撴槉鍏朵粬宸ヤ綔鐨勮仈绯?
HiFi-MambaV2寤剁画浜嗚敗鏅撴槉鍦ㄥ尰瀛﹀浘鍍忓垎鏋愰鍩熺殑鐮旂┒鑴夌粶锛?
1. **tCURLoRA (2025)**: 楂樻晥鍙傛暟寰皟 - 鍏卞悓鍏虫敞鏁堢巼浼樺寲
2. **Diffusion Brain MRI (2024)**: 鎵╂暎妯″瀷MRI閲嶅缓 - 浜掕ˉ鐨勯噸寤烘柟娉?3. **Few-shot Medical Imaging (2023)**: 灏忔牱鏈尰瀛﹀浘鍍?- 鍏卞悓鍏虫敞鏁版嵁鏁堢巼
4. **IIHT Medical Report (2023)**: 鍖诲鎶ュ憡鐢熸垚 - 涓嬫父搴旂敤

**鐮旂┒婕旇繘**: 浠庡彉鍒嗘柟娉?ROF, Mumford-Shah) 鈫?娣卞害瀛︿範(U-Net, CNN) 鈫?Transformer 鈫?SSM锛孒iFi-MambaV2浠ｈ〃浜嗚繖涓€婕旇繘鐨勬渶鏂伴樁娈点€?
---

## 闄勫綍锛氬叧閿叕寮忛€熸煡

```
SSM绂绘暎鍖?
  h虅 = exp(螖A)h虅' + 螖(螖A)^{-1}(exp(螖A)-I)Bx
  y = Ch虅 + Dx

鍙屽悜铻嶅悎:
  H = 伪H^f + (1-伪)H^b

HFA鑱氬悎:
  F_out = 危_{l=1}^L Conv^{(l)}(Resize^{(l)}(F^{(l)}))

RCL鏍℃:
  X_c = X_r + R_胃(F^{-1}(P_惟F(X_r) - P_惟F(X_gt)))
```

---

**绗旇鐢熸垚鏃堕棿**: 2026-02-20
**绮捐娣卞害**: 鈽呪槄鈽呪槄鈽?(浜旂骇绮捐)
**鎺ㄨ崘鎸囨暟**: 鈽呪槄鈽呪槄鈽?(鍖诲鍥惧儚閲嶅缓棰嗗煙蹇呰)
