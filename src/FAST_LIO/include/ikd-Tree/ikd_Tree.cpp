#include "ikd_Tree.h"

/*
Description: ikd-Tree: an incremental k-d tree for robotic applications 
Author: Yixi Cai
email: yixicai@connect.hku.hk
*/

template <typename PointType>
KD_TREE<PointType>::KD_TREE(float delete_param, float balance_param, float box_length) {
    delete_criterion_param = delete_param;
    balance_criterion_param = balance_param;
    downsample_size = box_length;
    Rebuild_Logger.clear();           
    termination_flag = false;
    start_thread();
}

template <typename PointType>
KD_TREE<PointType>::~KD_TREE()
{
    stop_thread();
    Delete_Storage_Disabled = true;
    delete_tree_nodes(&Root_Node);
    PointVector ().swap(PCL_Storage);
    Rebuild_Logger.clear();           
}

template <typename PointType>
void KD_TREE<PointType>::Set_delete_criterion_param(float delete_param){
    delete_criterion_param = delete_param;
}

template <typename PointType>
void KD_TREE<PointType>::Set_balance_criterion_param(float balance_param){
    balance_criterion_param = balance_param;
}

template <typename PointType>
void KD_TREE<PointType>::set_downsample_param(float downsample_param){
    downsample_size = downsample_param;
}

template <typename PointType>
void KD_TREE<PointType>::InitializeKDTree(float delete_param, float balance_param, float box_length){
    Set_delete_criterion_param(delete_param);
    Set_balance_criterion_param(balance_param);
    set_downsample_param(box_length);
}

template <typename PointType>
void KD_TREE<PointType>::InitTreeNode(KD_TREE_NODE * root){
    root->point.x = 0.0f;
    root->point.y = 0.0f;
    root->point.z = 0.0f;       
    root->node_range_x[0] = 0.0f;
    root->node_range_x[1] = 0.0f;
    root->node_range_y[0] = 0.0f;
    root->node_range_y[1] = 0.0f;    
    root->node_range_z[0] = 0.0f;
    root->node_range_z[1] = 0.0f;     
    root->division_axis = 0;
    root->father_ptr = nullptr;
    root->left_son_ptr = nullptr;
    root->right_son_ptr = nullptr;
    root->TreeSize = 0;
    root->invalid_point_num = 0;
    root->down_del_num = 0;
    root->point_deleted = false;
    root->tree_deleted = false;
    root->need_push_down_to_left = false;
    root->need_push_down_to_right = false;
    root->point_downsample_deleted = false;
    root->working_flag = false;
    pthread_mutex_init(&(root->push_down_mutex_lock),NULL);
}   

template <typename PointType>
int KD_TREE<PointType>::size(){
    int s = 0;
    if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != Root_Node){
        if (Root_Node != nullptr) {
            return Root_Node->TreeSize;
        } else {
            return 0;
        }
    } else {
        if (!pthread_mutex_trylock(&working_flag_mutex)){
            s = Root_Node->TreeSize;
            pthread_mutex_unlock(&working_flag_mutex);
            return s;
        } else {
            return Treesize_tmp;
        }
    }
}

template <typename PointType>
BoxPointType KD_TREE<PointType>::tree_range(){
    BoxPointType range;
    if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != Root_Node){
        if (Root_Node != nullptr) {
            range.vertex_min[0] = Root_Node->node_range_x[0];
            range.vertex_min[1] = Root_Node->node_range_y[0];
            range.vertex_min[2] = Root_Node->node_range_z[0];
            range.vertex_max[0] = Root_Node->node_range_x[1];
            range.vertex_max[1] = Root_Node->node_range_y[1];
            range.vertex_max[2] = Root_Node->node_range_z[1];
        } else {
            memset(&range, 0, sizeof(range));
        }
    } else {
        if (!pthread_mutex_trylock(&working_flag_mutex)){
            range.vertex_min[0] = Root_Node->node_range_x[0];
            range.vertex_min[1] = Root_Node->node_range_y[0];
            range.vertex_min[2] = Root_Node->node_range_z[0];
            range.vertex_max[0] = Root_Node->node_range_x[1];
            range.vertex_max[1] = Root_Node->node_range_y[1];
            range.vertex_max[2] = Root_Node->node_range_z[1];
            pthread_mutex_unlock(&working_flag_mutex);
        } else {
            memset(&range, 0, sizeof(range));
        }
    }
    return range;
}

template <typename PointType>
int KD_TREE<PointType>::validnum(){
    int s = 0;
    if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != Root_Node){
        if (Root_Node != nullptr)
            return (Root_Node->TreeSize - Root_Node->invalid_point_num);
        else 
            return 0;
    } else {
        if (!pthread_mutex_trylock(&working_flag_mutex)){
            s = Root_Node->TreeSize-Root_Node->invalid_point_num;
            pthread_mutex_unlock(&working_flag_mutex);
            return s;
        } else {
            return -1;
        }
    }
}

template <typename PointType>
void KD_TREE<PointType>::root_alpha(float &alpha_bal, float &alpha_del){
    if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != Root_Node){
        alpha_bal = Root_Node->alpha_bal;
        alpha_del = Root_Node->alpha_del;
        return;
    } else {
        if (!pthread_mutex_trylock(&working_flag_mutex)){
            alpha_bal = Root_Node->alpha_bal;
            alpha_del = Root_Node->alpha_del;
            pthread_mutex_unlock(&working_flag_mutex);
            return;
        } else {
            alpha_bal = alpha_bal_tmp;
            alpha_del = alpha_del_tmp;      
            return;
        }
    }    
}

template <typename PointType>
void KD_TREE<PointType>::start_thread(){
    pthread_mutex_init(&termination_flag_mutex_lock, NULL);   
    pthread_mutex_init(&rebuild_ptr_mutex_lock, NULL);     
    pthread_mutex_init(&rebuild_logger_mutex_lock, NULL);
    pthread_mutex_init(&points_deleted_rebuild_mutex_lock, NULL); 
    pthread_mutex_init(&working_flag_mutex, NULL);
    pthread_mutex_init(&search_flag_mutex, NULL);
    pthread_create(&rebuild_thread, NULL, multi_thread_ptr, (void*) this);
    printf("Multi thread started \n");    
}

template <typename PointType>
void KD_TREE<PointType>::stop_thread(){
    pthread_mutex_lock(&termination_flag_mutex_lock);
    termination_flag = true;
    pthread_mutex_unlock(&termination_flag_mutex_lock);
    if (rebuild_thread) pthread_join(rebuild_thread, NULL);
    pthread_mutex_destroy(&termination_flag_mutex_lock);
    pthread_mutex_destroy(&rebuild_logger_mutex_lock);
    pthread_mutex_destroy(&rebuild_ptr_mutex_lock);
    pthread_mutex_destroy(&points_deleted_rebuild_mutex_lock);
    pthread_mutex_destroy(&working_flag_mutex);
    pthread_mutex_destroy(&search_flag_mutex);     
}

template <typename PointType>
void * KD_TREE<PointType>::multi_thread_ptr(void * arg){
    KD_TREE * handle = (KD_TREE*) arg;
    handle->multi_thread_rebuild();
    return nullptr;    
}
// 构建ikd tree后，该线程一直在运行
template <typename PointType>
void KD_TREE<PointType>::multi_thread_rebuild(){
 bool terminated = false;
    // 新建虚拟头结点和重建的树的新节点
    KD_TREE_NODE * father_ptr, ** new_node_ptr;
    pthread_mutex_lock(&termination_flag_mutex_lock);
    // KD树初始化时termination_flag被设为false，所以后面的while循环是一直进行的
    terminated = termination_flag;
    pthread_mutex_unlock(&termination_flag_mutex_lock);
    while (!terminated){
        // The second thread will lock all incremental updates (i.e., points insert, re-insert,
        // and delete) but not queries on this sub-tree
        // 锁定了插入、重插入、删除功能，但不锁定查询功能
        pthread_mutex_lock(&rebuild_ptr_mutex_lock);
        pthread_mutex_lock(&working_flag_mutex);
        // Rebuild_Ptr是一个tree_node类型的指针的指针
        // 如果有需要重建的话，Rebuild_Ptr就是需要重建的子树的根节点
        if (Rebuild_Ptr != nullptr ){                    
            /* Traverse and copy */
            // 如果在重建某个子树的过程中，用户对该子树有新的修改操作，例如插入、删除、属性传递等，
            // 就会将这些操作记录到Rebuild_Logger中，等完成重建后再补作业
            if (!Rebuild_Logger.empty()){
                printf("\n\n\n\n\n\n\n\n\n\n\n ERROR!!! \n\n\n\n\n\n\n\n\n");
            }
            rebuild_flag = true;
            // 判断重建的子树的根节点是否为整棵树的实际根节点
            if (*Rebuild_Ptr == Root_Node) {
                // 如果是的话，要记录一些根节点特有的参数，避免重建过程中造成数据丢失，因为待会树的实际根节点会被替换
                Treesize_tmp = Root_Node->TreeSize;
                Validnum_tmp = Root_Node->TreeSize - Root_Node->invalid_point_num;
                alpha_bal_tmp = Root_Node->alpha_bal;
                alpha_del_tmp = Root_Node->alpha_del;
            }
            // 重建前子树的根节点
            KD_TREE_NODE * old_root_node = (*Rebuild_Ptr);
            // 记录该子树的父节点，重建完成后需要重新接驳回去                            
            father_ptr = (*Rebuild_Ptr)->father_ptr;  
            // 清空缓存区，待会用于存储子树内的所有点
            PointVector ().swap(Rebuild_PCL_Storage);
            // Lock Search 
            // 禁用查找功能
            pthread_mutex_lock(&search_flag_mutex);
            // 如果正在进行查找，则等待
            while (search_mutex_counter != 0){
                pthread_mutex_unlock(&search_flag_mutex);
                usleep(1);             
                pthread_mutex_lock(&search_flag_mutex);
            }
            search_mutex_counter = -1;
            pthread_mutex_unlock(&search_flag_mutex);
            // Lock deleted points cache
            // 禁用删除功能
            pthread_mutex_lock(&points_deleted_rebuild_mutex_lock);
            // 将树展平，将子树中的所有点存到Rebuild_PCL_Storage，注意输入参数中的`MULTI_THREAD_REC`是指定是否记录删除点，这些删除点估计是用来做debug
            flatten(*Rebuild_Ptr, Rebuild_PCL_Storage, MULTI_THREAD_REC);
            // Unlock deleted points cache
            // 记录完待重建子树的点，解锁删除功能
            pthread_mutex_unlock(&points_deleted_rebuild_mutex_lock);
            // Unlock Search
            // 解锁搜索功能，将search_mutex_counter归零
            pthread_mutex_lock(&search_flag_mutex);
            search_mutex_counter = 0;
            pthread_mutex_unlock(&search_flag_mutex);              
            pthread_mutex_unlock(&working_flag_mutex);   
            /* Rebuild and update missed operations*/
            Operation_Logger_Type Operation;
            KD_TREE_NODE * new_root_node = nullptr;  
            // 如果展平的树的节点大于0，则可以开始重建
            if (int(Rebuild_PCL_Storage.size()) > 0){
                // 重建树，树的根节点为new_root_node
                BuildTree(&new_root_node, 0, Rebuild_PCL_Storage.size()-1, Rebuild_PCL_Storage);
                // Rebuild has been done. Updates the blocked operations into the new tree
                pthread_mutex_lock(&working_flag_mutex);
                pthread_mutex_lock(&rebuild_logger_mutex_lock);
                int tmp_counter = 0;
                // 补作业：重建树完成后，将中途的更新操作（插入、重插入、删除）操作补充
                while (!Rebuild_Logger.empty()){
                    // 从Rebuild_Logger读取操作，是插入？重插入？还是删除？
                    Operation = Rebuild_Logger.front();
                    max_queue_size = max(max_queue_size, Rebuild_Logger.size());
                    Rebuild_Logger.pop();
                    pthread_mutex_unlock(&rebuild_logger_mutex_lock);                  
                    pthread_mutex_unlock(&working_flag_mutex);
                    // 根据Operation中的枚举变量指定在新的树中操作
                    run_operation(&new_root_node, Operation);
                    tmp_counter ++;
                    if (tmp_counter % 10 == 0) usleep(1);
                    pthread_mutex_lock(&working_flag_mutex);
                    pthread_mutex_lock(&rebuild_logger_mutex_lock);               
                }   
               pthread_mutex_unlock(&rebuild_logger_mutex_lock);
            }  
            /* Replace to original tree*/          
            // pthread_mutex_lock(&working_flag_mutex);
            // 等待查找功能的结束
            pthread_mutex_lock(&search_flag_mutex);
            while (search_mutex_counter != 0){
                pthread_mutex_unlock(&search_flag_mutex);
                usleep(1);             
                pthread_mutex_lock(&search_flag_mutex);
            }
            search_mutex_counter = -1;
            pthread_mutex_unlock(&search_flag_mutex);
            // 用新树替换原来的树
            if (father_ptr->left_son_ptr == *Rebuild_Ptr) {
                father_ptr->left_son_ptr = new_root_node;
            } else if (father_ptr->right_son_ptr == *Rebuild_Ptr){             
                father_ptr->right_son_ptr = new_root_node;
            } else {
                throw "Error: Father ptr incompatible with current node\n";
            }
            // 给新的子树重新配置虚拟头结点
            if (new_root_node != nullptr) new_root_node->father_ptr = father_ptr;
            // 将新的头结点赋给重建的树
            (*Rebuild_Ptr) = new_root_node;
            // 重新配置树的属性
            int valid_old = old_root_node->TreeSize-old_root_node->invalid_point_num;
            int valid_new = new_root_node->TreeSize-new_root_node->invalid_point_num;
            if (father_ptr == STATIC_ROOT_NODE) Root_Node = STATIC_ROOT_NODE->left_son_ptr;
            KD_TREE_NODE * update_root = *Rebuild_Ptr;
            // 将子树属性上拉到树的根节点
            while (update_root != nullptr && update_root != Root_Node){
                update_root = update_root->father_ptr;
                if (update_root->working_flag) break;
                if (update_root == update_root->father_ptr->left_son_ptr && update_root->father_ptr->need_push_down_to_left) break;
                if (update_root == update_root->father_ptr->right_son_ptr && update_root->father_ptr->need_push_down_to_right) break;
                Update(update_root);
            }
            pthread_mutex_lock(&search_flag_mutex);
            search_mutex_counter = 0;
            pthread_mutex_unlock(&search_flag_mutex);
            Rebuild_Ptr = nullptr;
            pthread_mutex_unlock(&working_flag_mutex);
            rebuild_flag = false;                     
            /* Delete discarded tree nodes */
            // 释放旧的子树的内存，前序遍历逐个释放
            delete_tree_nodes(&old_root_node);
        
        } else { // 不需要重建
            pthread_mutex_unlock(&working_flag_mutex);             
        }
        pthread_mutex_unlock(&rebuild_ptr_mutex_lock);         
        pthread_mutex_lock(&termination_flag_mutex_lock);
        terminated = termination_flag;
        pthread_mutex_unlock(&termination_flag_mutex_lock);
        // 重建树线程的运行周期
        usleep(100); 
    }
    printf("Rebuild thread terminated normally\n");   
}

template <typename PointType>
void KD_TREE<PointType>::run_operation(KD_TREE_NODE ** root, Operation_Logger_Type operation){
    switch (operation.op)
    {
    case ADD_POINT:      
        Add_by_point(root, operation.point, false, (*root)->division_axis);          
        break;
    case ADD_BOX:
        Add_by_range(root, operation.boxpoint, false);
        break;
    case DELETE_POINT:
        Delete_by_point(root, operation.point, false);
        break;
    case DELETE_BOX:
        Delete_by_range(root, operation.boxpoint, false, false);
        break;
    case DOWNSAMPLE_DELETE:
        Delete_by_range(root, operation.boxpoint, false, true);
        break;
    case PUSH_DOWN:
        (*root)->tree_downsample_deleted |= operation.tree_downsample_deleted;
        (*root)->point_downsample_deleted |= operation.tree_downsample_deleted;
        (*root)->tree_deleted = operation.tree_deleted || (*root)->tree_downsample_deleted;
        (*root)->point_deleted = (*root)->tree_deleted || (*root)->point_downsample_deleted;
        if (operation.tree_downsample_deleted) (*root)->down_del_num = (*root)->TreeSize;
        if (operation.tree_deleted) (*root)->invalid_point_num = (*root)->TreeSize;
            else (*root)->invalid_point_num = (*root)->down_del_num;
        (*root)->need_push_down_to_left = true;
        (*root)->need_push_down_to_right = true;     
        break;
    default:
        break;
    }
}

template <typename PointType>
void KD_TREE<PointType>::Build(PointVector point_cloud){
    if (Root_Node != nullptr){
        delete_tree_nodes(&Root_Node);//如果有root需要删除以前的root
    }
    if (point_cloud.size() == 0) return;
    STATIC_ROOT_NODE = new KD_TREE_NODE;// 在堆上new一个节点作为树的根节点，不过它是一个虚节点
    InitTreeNode(STATIC_ROOT_NODE); //初始化当前节点的属性
    // 注意当前左子树为空指针，为了传入函数，用了取址符号&，因此传入的是左子树指针的地址，也就是指针的指针
    BuildTree(&STATIC_ROOT_NODE->left_son_ptr, 0, point_cloud.size()-1, point_cloud);//建立kdtree，这里直接从其左子树开始构建
    Update(STATIC_ROOT_NODE);//更新root的属性
    STATIC_ROOT_NODE->TreeSize = 0;
    Root_Node = STATIC_ROOT_NODE->left_son_ptr;// 将左子树直接替换根节点，因为将虚节点的左节点设为了第一个节点
}

template <typename PointType>
void KD_TREE<PointType>::Nearest_Search(PointType point, int k_nearest, PointVector& Nearest_Points, vector<float> & Point_Distance, double max_dist){   
    MANUAL_HEAP q(2*k_nearest);
    q.clear();
    vector<float> ().swap(Point_Distance);
    if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != Root_Node){
        Search(Root_Node, k_nearest, point, q, max_dist);
    } else {
        pthread_mutex_lock(&search_flag_mutex);
        while (search_mutex_counter == -1)
        {
            pthread_mutex_unlock(&search_flag_mutex);
            usleep(1);
            pthread_mutex_lock(&search_flag_mutex);
        }
        search_mutex_counter += 1;
        pthread_mutex_unlock(&search_flag_mutex);  
        Search(Root_Node, k_nearest, point, q, max_dist);  
        pthread_mutex_lock(&search_flag_mutex);
        search_mutex_counter -= 1;
        pthread_mutex_unlock(&search_flag_mutex);      
    }
    int k_found = min(k_nearest,int(q.size()));
    PointVector ().swap(Nearest_Points);
    vector<float> ().swap(Point_Distance);
    for (int i=0;i < k_found;i++){
        Nearest_Points.insert(Nearest_Points.begin(), q.top().point);
        Point_Distance.insert(Point_Distance.begin(), q.top().dist);
        q.pop();
    }
    return;
}

template <typename PointType>
void KD_TREE<PointType>::Box_Search(const BoxPointType &Box_of_Point, PointVector &Storage)
{
    Storage.clear();
    Search_by_range(Root_Node, Box_of_Point, Storage);
}

template <typename PointType>
void KD_TREE<PointType>::Radius_Search(PointType point, const float radius, PointVector &Storage)
{
    Storage.clear();
    Search_by_radius(Root_Node, point, radius, Storage);
}

template <typename PointType>
int KD_TREE<PointType>::Add_Points(PointVector & PointToAdd, bool downsample_on){
    int NewPointSize = PointToAdd.size();// 需要插入的点的个数
    int tree_size = size();// 树的节点个数
    BoxPointType Box_of_Point;
    PointType downsample_result, mid_point;
    bool downsample_switch = downsample_on && DOWNSAMPLE_SWITCH;// 根据输入参数判断是否需要进行降采样
    float min_dist, tmp_dist;
    int tmp_counter = 0;
    // 获得插入点所在的Voxel，计算Voxel的几何中心点（将来只保留最接近中心点的point）
    for (int i=0; i<PointToAdd.size();i++){
        if (downsample_switch){// 判断是否需要降采样
        // 计算该点所属的体素
            Box_of_Point.vertex_min[0] = floor(PointToAdd[i].x/downsample_size)*downsample_size;
            Box_of_Point.vertex_max[0] = Box_of_Point.vertex_min[0]+downsample_size;
            Box_of_Point.vertex_min[1] = floor(PointToAdd[i].y/downsample_size)*downsample_size;
            Box_of_Point.vertex_max[1] = Box_of_Point.vertex_min[1]+downsample_size; 
            Box_of_Point.vertex_min[2] = floor(PointToAdd[i].z/downsample_size)*downsample_size;
            Box_of_Point.vertex_max[2] = Box_of_Point.vertex_min[2]+downsample_size;   
            mid_point.x = Box_of_Point.vertex_min[0] + (Box_of_Point.vertex_max[0]-Box_of_Point.vertex_min[0])/2.0;// 计算该体素的中心点坐标
            mid_point.y = Box_of_Point.vertex_min[1] + (Box_of_Point.vertex_max[1]-Box_of_Point.vertex_min[1])/2.0;// 直接加0.5*downsample_size不就得了吗
            mid_point.z = Box_of_Point.vertex_min[2] + (Box_of_Point.vertex_max[2]-Box_of_Point.vertex_min[2])/2.0;
            PointVector ().swap(Downsample_Storage);// 清空降采样缓存
            Search_by_range(Root_Node, Box_of_Point, Downsample_Storage);// 在树中搜索最近邻点，且要求点都在Box中，将点存于Downsample_Storage
            min_dist = calc_dist(PointToAdd[i],mid_point);// 计算当前点与体素中心的距离
            downsample_result = PointToAdd[i];       
            // 遍历体素内的所有点,寻找近邻点中最接近体素中心的点          
            for (int index = 0; index < Downsample_Storage.size(); index++){
                // 计算当前近邻点与体素中心的距离
                tmp_dist = calc_dist(Downsample_Storage[index], mid_point);
                 // 比较两个距离，判断是否需要添加该点
                if (tmp_dist < min_dist){
                    min_dist = tmp_dist;
                    downsample_result = Downsample_Storage[index];//距离体素中心最近的临近点
                }
            }
            // 如果不需要重建树
            // 如果当前没有re-balancing任务，也即没有并行线程，则直接执行`BoxDelete`和`插入一个点`
            if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != Root_Node){  
                // 如果近邻点不止1个，或者当前点与原有的点很接近
                if (Downsample_Storage.size() > 1 || same_point(PointToAdd[i], downsample_result)){//same_point计算三轴距离并与阈值比较
                    // 删除体素内的原有点，然后插入当前点
                    if (Downsample_Storage.size() > 0) Delete_by_range(&Root_Node, Box_of_Point, true, true);
                    Add_by_point(&Root_Node, downsample_result, true, Root_Node->division_axis);
                    tmp_counter ++;                      
                }
            } else {// 需要重建树 如果有re-balancing任务在并行运行，
                    //在对当前tree执行`BoxDelete`和`插入一个点`之外，还需要把这些操作缓存到logger里
                if (Downsample_Storage.size() > 1 || same_point(PointToAdd[i], downsample_result)){
                    Operation_Logger_Type  operation_delete, operation;
                    operation_delete.boxpoint = Box_of_Point; // 记录待删除的点
                    operation_delete.op = DOWNSAMPLE_DELETE;
                    operation.point = downsample_result;// 记录待插入的新点
                    operation.op = ADD_POINT;
                    pthread_mutex_lock(&working_flag_mutex);// 删除体素内的点
                    if (Downsample_Storage.size() > 0) Delete_by_range(&Root_Node, Box_of_Point, false , true);                                      
                    Add_by_point(&Root_Node, downsample_result, false, Root_Node->division_axis);
                    tmp_counter ++;
                    if (rebuild_flag){
                        pthread_mutex_lock(&rebuild_logger_mutex_lock);
                        if (Downsample_Storage.size() > 0) Rebuild_Logger.push(operation_delete);
                        Rebuild_Logger.push(operation);
                        pthread_mutex_unlock(&rebuild_logger_mutex_lock);
                    }
                    pthread_mutex_unlock(&working_flag_mutex);
                };
            }
        } else {// 不需要降采样
        // 不需要重建 或 带重建的子树不是当前整个树（Root_Node）
            if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != Root_Node){
                Add_by_point(&Root_Node, PointToAdd[i], true, Root_Node->division_axis); // 直接插入单个点  
            } else {// 如果有并行re-balancing任务，还需额外把当前操作放入logger缓存，等重建完成后补作业
                Operation_Logger_Type operation;
                operation.point = PointToAdd[i];
                operation.op = ADD_POINT;                
                pthread_mutex_lock(&working_flag_mutex);
                Add_by_point(&Root_Node, PointToAdd[i], false, Root_Node->division_axis);
                if (rebuild_flag){// 记录本次插入点的操作
                    pthread_mutex_lock(&rebuild_logger_mutex_lock);
                    Rebuild_Logger.push(operation);
                    pthread_mutex_unlock(&rebuild_logger_mutex_lock);
                }
                pthread_mutex_unlock(&working_flag_mutex);       
            }
        }
    }
    return tmp_counter;
}

template <typename PointType>
void KD_TREE<PointType>::Add_Point_Boxes(vector<BoxPointType> & BoxPoints){     
    for (int i=0;i < BoxPoints.size();i++){
        if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != Root_Node){
            Add_by_range(&Root_Node ,BoxPoints[i], true);
        } else {
            Operation_Logger_Type operation;
            operation.boxpoint = BoxPoints[i];
            operation.op = ADD_BOX;
            pthread_mutex_lock(&working_flag_mutex);
            Add_by_range(&Root_Node ,BoxPoints[i], false);
            if (rebuild_flag){
                pthread_mutex_lock(&rebuild_logger_mutex_lock);
                Rebuild_Logger.push(operation);
                pthread_mutex_unlock(&rebuild_logger_mutex_lock);
            }               
            pthread_mutex_unlock(&working_flag_mutex);
        }    
    } 
    return;
}

template <typename PointType>
void KD_TREE<PointType>::Delete_Points(PointVector & PointToDel){        
    for (int i=0;i<PointToDel.size();i++){
        if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != Root_Node){               
            Delete_by_point(&Root_Node, PointToDel[i], true);
        } else {
            Operation_Logger_Type operation;
            operation.point = PointToDel[i];
            operation.op = DELETE_POINT;
            pthread_mutex_lock(&working_flag_mutex);        
            Delete_by_point(&Root_Node, PointToDel[i], false);
            if (rebuild_flag){
                pthread_mutex_lock(&rebuild_logger_mutex_lock);
                Rebuild_Logger.push(operation);
                pthread_mutex_unlock(&rebuild_logger_mutex_lock);
            }
            pthread_mutex_unlock(&working_flag_mutex);
        }      
    }      
    return;
}

template <typename PointType>
int KD_TREE<PointType>::Delete_Point_Boxes(vector<BoxPointType> & BoxPoints){
    int tmp_counter = 0;
    for (int i=0;i < BoxPoints.size();i++){ 
        if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != Root_Node){               
            tmp_counter += Delete_by_range(&Root_Node ,BoxPoints[i], true, false);
        } else {
            Operation_Logger_Type operation;
            operation.boxpoint = BoxPoints[i];
            operation.op = DELETE_BOX;     
            pthread_mutex_lock(&working_flag_mutex); 
            tmp_counter += Delete_by_range(&Root_Node ,BoxPoints[i], false, false);
            if (rebuild_flag){
                pthread_mutex_lock(&rebuild_logger_mutex_lock);
                Rebuild_Logger.push(operation);
                pthread_mutex_unlock(&rebuild_logger_mutex_lock);
            }                
            pthread_mutex_unlock(&working_flag_mutex);
        }
    } 
    return tmp_counter;
}

template <typename PointType>
void KD_TREE<PointType>::acquire_removed_points(PointVector & removed_points){
    pthread_mutex_lock(&points_deleted_rebuild_mutex_lock); 
    for (int i = 0; i < Points_deleted.size();i++){
        removed_points.push_back(Points_deleted[i]);
    }
    for (int i = 0; i < Multithread_Points_deleted.size();i++){
        removed_points.push_back(Multithread_Points_deleted[i]);
    }
    Points_deleted.clear();
    Multithread_Points_deleted.clear();
    pthread_mutex_unlock(&points_deleted_rebuild_mutex_lock);   
    return;
}
// 工具属性，构建新的(sub)tree；这里的双指针形参很重要，能够允许我们传入一个空指针。
// 建树的过程与构建有序二叉树的过程类似，取中间的节点作为节点值进行递归
// 从Storage中取出第l到第r个点用于构建树
template <typename PointType>
void KD_TREE<PointType>::BuildTree(KD_TREE_NODE ** root, int l, int r, PointVector & Storage){
    if (l>r) return;
    *root = new KD_TREE_NODE;
    InitTreeNode(*root);
    int mid = (l+r)>>1;//计算l+r的中点
    int div_axis = 0;
    int i;
    // Find the best division Axis也即分布最分散的那个轴，或者说最大值减最小值之差最大的那个轴
    float min_value[3] = {INFINITY, INFINITY, INFINITY};
    float max_value[3] = {-INFINITY, -INFINITY, -INFINITY};
    float dim_range[3] = {0,0,0};
    for (i=l;i<=r;i++){//循环所有点找最大最小值
        min_value[0] = min(min_value[0], Storage[i].x);
        min_value[1] = min(min_value[1], Storage[i].y);
        min_value[2] = min(min_value[2], Storage[i].z);
        max_value[0] = max(max_value[0], Storage[i].x);
        max_value[1] = max(max_value[1], Storage[i].y);
        max_value[2] = max(max_value[2], Storage[i].z);
    }
    // Select the longest dimension as division axis 范围最大的当作分割轴
    for (i=0;i<3;i++) dim_range[i] = max_value[i] - min_value[i];
    for (i=1;i<3;i++) if (dim_range[i] > dim_range[div_axis]) div_axis = i;
    // Divide by the division axis and recursively build.
    /*********************************************
     * nth_element(first, nth, last, compare)
     求[first, last]这个区间中第n大小的元素，如果参数加入了compare函数，就按compare函数的方式比较。
     这一段就是找切割维度的中位数

     nth_element(a,a+k,a+n)，函数只是把下标为k的元素放在了正确位置，对其它元素并没有排序，
     当然k左边元素都小于等于它，右边元素都大于等于它，所以可以利用这个函数快速定位某个元素。
    ******************************************************/
    (*root)->division_axis = div_axis;// 给该节点标记划分轴
    switch (div_axis)
    {
    case 0:
        nth_element(begin(Storage)+l, begin(Storage)+mid, begin(Storage)+r+1, point_cmp_x);
        break;
    case 1:
        nth_element(begin(Storage)+l, begin(Storage)+mid, begin(Storage)+r+1, point_cmp_y);
        break;
    case 2:
        nth_element(begin(Storage)+l, begin(Storage)+mid, begin(Storage)+r+1, point_cmp_z);
        break;
    default:
        nth_element(begin(Storage)+l, begin(Storage)+mid, begin(Storage)+r+1, point_cmp_x);
        break;
    }  
    (*root)->point = Storage[mid]; // 该节点的点
    KD_TREE_NODE * left_son = nullptr, * right_son = nullptr;
    // 递归构建整个tree（自上而下）。
    // 这就是一般的用队列建树的流程， 属于是二叉树前序遍历过程
    BuildTree(&left_son, l, mid-1, Storage);//左右子节点
    BuildTree(&right_son, mid+1, r, Storage);  
    (*root)->left_son_ptr = left_son;// 标记该节点的左右子树
    (*root)->right_son_ptr = right_son;
    Update((*root));//更新node的属性
    return;
}
// 重建树
template <typename PointType>
void KD_TREE<PointType>::Rebuild(KD_TREE_NODE ** root){    
    KD_TREE_NODE * father_ptr;
    // 如果树的节点数大于阈值，则启动多线程进行重建
    if ((*root)->TreeSize >= Multi_Thread_Rebuild_Point_Num) { 
        // pthread_mutex_trylock() 是 pthread_mutex_lock() 的非阻塞版本。
        // 如果 mutex 所引用的互斥对象当前被任何线程（包括当前线程）锁定，则将立即返
        // 回该调用。否则，该互斥锁将处于锁定状态，调用线程是其属主。
        if (!pthread_mutex_trylock(&rebuild_ptr_mutex_lock)){ // true：成功加锁，fasle：其他线程已上锁
        // 如果重建树的指针为空，或者当前子树的尺寸比正在重建的子树大，则将重建树的根节点设为当前根节点
            if (Rebuild_Ptr == nullptr || ((*root)->TreeSize > (*Rebuild_Ptr)->TreeSize)) {
                Rebuild_Ptr = root;          
            }
            pthread_mutex_unlock(&rebuild_ptr_mutex_lock);
            // 解锁之后即开始启动多线程重建
        }
    } else { // 单线程重建
        // 暂存虚拟父节点
        father_ptr = (*root)->father_ptr;
        int size_rec = (*root)->TreeSize;
        PCL_Storage.clear();
        // 将当前子树的节点展平，同时做被删除点的记录
        flatten(*root, PCL_Storage, DELETE_POINTS_REC);
        delete_tree_nodes(root);// 释放被删除点的内存空间
        BuildTree(root, 0, PCL_Storage.size()-1, PCL_Storage);
        if (*root != nullptr) (*root)->father_ptr = father_ptr;
        if (*root == Root_Node) STATIC_ROOT_NODE->left_son_ptr = *root;
    } 
    return;
}
// box-wise delete 工具属性，从根节点开始向下递归搜索，删除（仅标记）所有被Box包含的节点。
template <typename PointType>
int KD_TREE<PointType>::Delete_by_range(KD_TREE_NODE ** root,  BoxPointType boxpoint, bool allow_rebuild, bool is_downsample){   
    if ((*root) == nullptr || (*root)->tree_deleted) return 0;
    (*root)->working_flag = true;// 下拉属性
    Push_Down(*root); // 向下更新信息。
    int tmp_counter = 0;
    // 当且仅当两个空间有交叉时，才有继续的必要。
    // 前三行是判断给定的Box范围是否与树有交集，没有的话就不用搜索了 
    if (boxpoint.vertex_max[0] <= (*root)->node_range_x[0] || boxpoint.vertex_min[0] > (*root)->node_range_x[1]) return 0;
    if (boxpoint.vertex_max[1] <= (*root)->node_range_y[0] || boxpoint.vertex_min[1] > (*root)->node_range_y[1]) return 0;
    if (boxpoint.vertex_max[2] <= (*root)->node_range_z[0] || boxpoint.vertex_min[2] > (*root)->node_range_z[1]) return 0;
    // 当Box完全包含了节点所张成的空间时，把该节点subtree上的所有点标记删除。
    if (boxpoint.vertex_min[0] <= (*root)->node_range_x[0] && boxpoint.vertex_max[0] > (*root)->node_range_x[1] && boxpoint.vertex_min[1] <= (*root)->node_range_y[0] && boxpoint.vertex_max[1] > (*root)->node_range_y[1] && boxpoint.vertex_min[2] <= (*root)->node_range_z[0] && boxpoint.vertex_max[2] > (*root)->node_range_z[1]){
        (*root)->tree_deleted = true;// 整棵树的属性都标记为删除
        (*root)->point_deleted = true;
        (*root)->need_push_down_to_left = true;// 标记下，下次需要下拉属性
        (*root)->need_push_down_to_right = true;
        tmp_counter = (*root)->TreeSize - (*root)->invalid_point_num;// 当前子树需要被删除的点的个数
        (*root)->invalid_point_num = (*root)->TreeSize;
        if (is_downsample){
            (*root)->tree_downsample_deleted = true;
            (*root)->point_downsample_deleted = true;
            (*root)->down_del_num = (*root)->TreeSize;
        }
        return tmp_counter;
    }
    // 如果当前节点的point被Box包含，标记删除该point。
    // 关键：当前节点是否被包含在给定box内，如果在的话就删除。因此box删除本质上也是逐个点地删除，从根节点向下收缩
    if (!(*root)->point_deleted && boxpoint.vertex_min[0] <= (*root)->point.x && boxpoint.vertex_max[0] > (*root)->point.x && boxpoint.vertex_min[1] <= (*root)->point.y && boxpoint.vertex_max[1] > (*root)->point.y && boxpoint.vertex_min[2] <= (*root)->point.z && boxpoint.vertex_max[2] > (*root)->point.z){
        (*root)->point_deleted = true;
        tmp_counter += 1;
        if (is_downsample) (*root)->point_downsample_deleted = true;
    }
    Operation_Logger_Type delete_box_log;
    struct timespec Timeout;    
    // 在左右子树中递归调用盒式删除 
    if (is_downsample) delete_box_log.op = DOWNSAMPLE_DELETE;
        else delete_box_log.op = DELETE_BOX;
    delete_box_log.boxpoint = boxpoint;
    // 左子树递归删除
    // 不需要多线程重建
    if ((Rebuild_Ptr == nullptr) || (*root)->left_son_ptr != *Rebuild_Ptr){
        // 递归到左子树，然后重复本步骤，判断左子节点的点是否包含在Box内，如果在的话就标记删除
        tmp_counter += Delete_by_range(&((*root)->left_son_ptr), boxpoint, allow_rebuild, is_downsample);
    } else { // 正在多线程重建
        pthread_mutex_lock(&working_flag_mutex);// 在旧子树里操作
        tmp_counter += Delete_by_range(&((*root)->left_son_ptr), boxpoint, false, is_downsample);
        // 完成重建后在新子树中补作业
        if (rebuild_flag){
            pthread_mutex_lock(&rebuild_logger_mutex_lock);
            Rebuild_Logger.push(delete_box_log);
            pthread_mutex_unlock(&rebuild_logger_mutex_lock);                 
        }
        pthread_mutex_unlock(&working_flag_mutex);
    }
    // 右子树递归删除
    if ((Rebuild_Ptr == nullptr) || (*root)->right_son_ptr != *Rebuild_Ptr){
        tmp_counter += Delete_by_range(&((*root)->right_son_ptr), boxpoint, allow_rebuild, is_downsample);
    } else {
        pthread_mutex_lock(&working_flag_mutex);
        tmp_counter += Delete_by_range(&((*root)->right_son_ptr), boxpoint, false, is_downsample);
        if (rebuild_flag){
            pthread_mutex_lock(&rebuild_logger_mutex_lock);
            Rebuild_Logger.push(delete_box_log);
            pthread_mutex_unlock(&rebuild_logger_mutex_lock);                 
        }
        pthread_mutex_unlock(&working_flag_mutex);
    }    
    // 属性上拉
    Update(*root);  // 更新当前节点信息（自下而上更新） 
    // 平衡性准则判断是否需要重建树  
    // 判断条件解读：重建树的指针的指针不为空，且重建树的指针为当前树的根节点，说明从上次重建后，就没有再更新这棵树的结构
    // 当前树的节点个数小于多线程重建的最小点阈值，则只需要单线程重建即可     
    if (Rebuild_Ptr != nullptr && *Rebuild_Ptr == *root && (*root)->TreeSize < Multi_Thread_Rebuild_Point_Num) Rebuild_Ptr = nullptr;  // 将Rebuild_Ptr设为空，即标记需要重建子树
    bool need_rebuild = allow_rebuild & Criterion_Check((*root));
    if (need_rebuild) Rebuild(root);
    if ((*root) != nullptr) (*root)->working_flag = false;
    return tmp_counter;
}

template <typename PointType>
void KD_TREE<PointType>::Delete_by_point(KD_TREE_NODE ** root, PointType point, bool allow_rebuild){   
    if ((*root) == nullptr || (*root)->tree_deleted) return;
    (*root)->working_flag = true;
    Push_Down(*root);
    if (same_point((*root)->point, point) && !(*root)->point_deleted) {          
        (*root)->point_deleted = true;
        (*root)->invalid_point_num += 1;
        if ((*root)->invalid_point_num == (*root)->TreeSize) (*root)->tree_deleted = true;    
        return;
    }
    Operation_Logger_Type delete_log;
    struct timespec Timeout;    
    delete_log.op = DELETE_POINT;
    delete_log.point = point;     
    if (((*root)->division_axis == 0 && point.x < (*root)->point.x) || ((*root)->division_axis == 1 && point.y < (*root)->point.y) || ((*root)->division_axis == 2 && point.z < (*root)->point.z)){           
        if ((Rebuild_Ptr == nullptr) || (*root)->left_son_ptr != *Rebuild_Ptr){          
            Delete_by_point(&(*root)->left_son_ptr, point, allow_rebuild);         
        } else {
            pthread_mutex_lock(&working_flag_mutex);
            Delete_by_point(&(*root)->left_son_ptr, point,false);
            if (rebuild_flag){
                pthread_mutex_lock(&rebuild_logger_mutex_lock);
                Rebuild_Logger.push(delete_log);
                pthread_mutex_unlock(&rebuild_logger_mutex_lock);                 
            }
            pthread_mutex_unlock(&working_flag_mutex);
        }
    } else {       
        if ((Rebuild_Ptr == nullptr) || (*root)->right_son_ptr != *Rebuild_Ptr){         
            Delete_by_point(&(*root)->right_son_ptr, point, allow_rebuild);         
        } else {
            pthread_mutex_lock(&working_flag_mutex); 
            Delete_by_point(&(*root)->right_son_ptr, point, false);
            if (rebuild_flag){
                pthread_mutex_lock(&rebuild_logger_mutex_lock);
                Rebuild_Logger.push(delete_log);
                pthread_mutex_unlock(&rebuild_logger_mutex_lock);                 
            }
            pthread_mutex_unlock(&working_flag_mutex);
        }        
    }
    Update(*root);
    if (Rebuild_Ptr != nullptr && *Rebuild_Ptr == *root && (*root)->TreeSize < Multi_Thread_Rebuild_Point_Num) Rebuild_Ptr = nullptr; 
    bool need_rebuild = allow_rebuild & Criterion_Check((*root));
    if (need_rebuild) Rebuild(root);
    if ((*root) != nullptr) (*root)->working_flag = false;   
    return;
}

template <typename PointType>
void KD_TREE<PointType>::Add_by_range(KD_TREE_NODE ** root, BoxPointType boxpoint, bool allow_rebuild){
    if ((*root) == nullptr) return;
    (*root)->working_flag = true;
    Push_Down(*root);       
    if (boxpoint.vertex_max[0] <= (*root)->node_range_x[0] || boxpoint.vertex_min[0] > (*root)->node_range_x[1]) return;
    if (boxpoint.vertex_max[1] <= (*root)->node_range_y[0] || boxpoint.vertex_min[1] > (*root)->node_range_y[1]) return;
    if (boxpoint.vertex_max[2] <= (*root)->node_range_z[0] || boxpoint.vertex_min[2] > (*root)->node_range_z[1]) return;
    if (boxpoint.vertex_min[0] <= (*root)->node_range_x[0] && boxpoint.vertex_max[0] > (*root)->node_range_x[1] && boxpoint.vertex_min[1] <= (*root)->node_range_y[0] && boxpoint.vertex_max[1]> (*root)->node_range_y[1] && boxpoint.vertex_min[2] <= (*root)->node_range_z[0] && boxpoint.vertex_max[2] > (*root)->node_range_z[1]){
        (*root)->tree_deleted = false || (*root)->tree_downsample_deleted;
        (*root)->point_deleted = false || (*root)->point_downsample_deleted;
        (*root)->need_push_down_to_left = true;
        (*root)->need_push_down_to_right = true;
        (*root)->invalid_point_num = (*root)->down_del_num; 
        return;
    }
    if (boxpoint.vertex_min[0] <= (*root)->point.x && boxpoint.vertex_max[0] > (*root)->point.x && boxpoint.vertex_min[1] <= (*root)->point.y && boxpoint.vertex_max[1] > (*root)->point.y && boxpoint.vertex_min[2] <= (*root)->point.z && boxpoint.vertex_max[2] > (*root)->point.z){
        (*root)->point_deleted = (*root)->point_downsample_deleted;
    }
    Operation_Logger_Type add_box_log;
    struct timespec Timeout;    
    add_box_log.op = ADD_BOX;
    add_box_log.boxpoint = boxpoint;
    if ((Rebuild_Ptr == nullptr) || (*root)->left_son_ptr != *Rebuild_Ptr){
        Add_by_range(&((*root)->left_son_ptr), boxpoint, allow_rebuild);
    } else {
        pthread_mutex_lock(&working_flag_mutex);
        Add_by_range(&((*root)->left_son_ptr), boxpoint, false);
        if (rebuild_flag){
            pthread_mutex_lock(&rebuild_logger_mutex_lock);
            Rebuild_Logger.push(add_box_log);
            pthread_mutex_unlock(&rebuild_logger_mutex_lock);                 
        }        
        pthread_mutex_unlock(&working_flag_mutex);
    }
    if ((Rebuild_Ptr == nullptr) || (*root)->right_son_ptr != *Rebuild_Ptr){
        Add_by_range(&((*root)->right_son_ptr), boxpoint, allow_rebuild);
    } else {
        pthread_mutex_lock(&working_flag_mutex);
        Add_by_range(&((*root)->right_son_ptr), boxpoint, false);
        if (rebuild_flag){
            pthread_mutex_lock(&rebuild_logger_mutex_lock);
            Rebuild_Logger.push(add_box_log);
            pthread_mutex_unlock(&rebuild_logger_mutex_lock);                 
        }
        pthread_mutex_unlock(&working_flag_mutex);
    }
    Update(*root);
    if (Rebuild_Ptr != nullptr && *Rebuild_Ptr == *root && (*root)->TreeSize < Multi_Thread_Rebuild_Point_Num) Rebuild_Ptr = nullptr; 
    bool need_rebuild = allow_rebuild & Criterion_Check((*root));
    if (need_rebuild) Rebuild(root);
    if ((*root) != nullptr) (*root)->working_flag = false;   
    return;
}
// 工具属性，单纯地往tree结构中插入新节点。
// 前序遍历的点级更新
// 在ik-d树上的点级更新以递归的方式实现，这类似于替罪羊k-d树。对于点级插入，
// 该算法递归地从根节点向下搜索，并将新点的除法轴上的坐标与存储在树节点上的点进行比较，
// 直到发现一个叶节点然后追加一个新的树节点。对于删除或重新插入一个点P，该算法会查找存
// 储该点P的树节点，并修改deleted属性。
template <typename PointType>
void KD_TREE<PointType>::Add_by_point(KD_TREE_NODE ** root, PointType point, bool allow_rebuild, int father_axis){     
    // 如果已经到达叶子节点，直接插入。
    // 如果当前节点为空的话，则新建节点
    if (*root == nullptr){
        *root = new KD_TREE_NODE; // 注意到root是指针的指针，所以不需要返回地址即可实现父节点指向当前新节点
        InitTreeNode(*root);
        (*root)->point = point;
        (*root)->division_axis = (father_axis + 1) % 3;// 标记该节点的划分轴
        Update(*root);// 将新的节点信息汇总给父节点
        return;
    }
    // `工作中`标志位置true，同步记录到Logger中。
    // 当前节点不为空，说明还未到叶子节点，则递归查找
    (*root)->working_flag = true;
    Operation_Logger_Type add_log; // 创建Operation_Logger_Type用于记录新插入的点，重构树的时候需要用到
    struct timespec Timeout;    
    add_log.op = ADD_POINT;
    add_log.point = point;
    Push_Down(*root); // 顺便将当前节点的状态信息下拉
    // 递归插入左子树。
    // 如果在左边则进行左边的比较然后添加，首先确定当前节点的划分轴，然后判断该点应该分到左子树还是右子树
    if (((*root)->division_axis == 0 && point.x < (*root)->point.x) || ((*root)->division_axis == 1 && point.y < (*root)->point.y) || ((*root)->division_axis == 2 && point.z < (*root)->point.z)){
         // 不需要重建 或 待重建的子树不是当前节点的左子树
        if ((Rebuild_Ptr == nullptr) || (*root)->left_son_ptr != *Rebuild_Ptr){          
            // 直接插入点
            Add_by_point(&(*root)->left_son_ptr, point, allow_rebuild, (*root)->division_axis);
        } else {// 该左子树正在重建的话，则将该操作记录到Rebuild_Logger，待重建完成后补作业
            // 线程互斥锁
            pthread_mutex_lock(&working_flag_mutex);
            Add_by_point(&(*root)->left_son_ptr, point, false,(*root)->division_axis);
            if (rebuild_flag){
                pthread_mutex_lock(&rebuild_logger_mutex_lock);
                Rebuild_Logger.push(add_log);
                pthread_mutex_unlock(&rebuild_logger_mutex_lock);                 
            }
            pthread_mutex_unlock(&working_flag_mutex);            
        }
    } else {  // 递归插入右子树。
        if ((Rebuild_Ptr == nullptr) || (*root)->right_son_ptr != *Rebuild_Ptr){         
            Add_by_point(&(*root)->right_son_ptr, point, allow_rebuild,(*root)->division_axis);
        } else {
            pthread_mutex_lock(&working_flag_mutex);
            Add_by_point(&(*root)->right_son_ptr, point, false,(*root)->division_axis);       
            if (rebuild_flag){
                pthread_mutex_lock(&rebuild_logger_mutex_lock);
                Rebuild_Logger.push(add_log);
                pthread_mutex_unlock(&rebuild_logger_mutex_lock);                 
            }
            pthread_mutex_unlock(&working_flag_mutex); 
        }
    }
    Update(*root);   // 更新当前节点信息（自下而上），并检查是否需要re-balancing。
    if (Rebuild_Ptr != nullptr && *Rebuild_Ptr == *root && (*root)->TreeSize < Multi_Thread_Rebuild_Point_Num) Rebuild_Ptr = nullptr; 
    // 判断子树是否平衡
    bool need_rebuild = allow_rebuild & Criterion_Check((*root));
    if (need_rebuild) Rebuild(root); // 检验后发现不平衡的话就需要重建该子树
    if ((*root) != nullptr) (*root)->working_flag = false;   
    return;
}

template <typename PointType>
void KD_TREE<PointType>::Search(KD_TREE_NODE * root, int k_nearest, PointType point, MANUAL_HEAP &q, double max_dist){
    if (root == nullptr || root->tree_deleted) return;   
    double cur_dist = calc_box_dist(root, point);
    double max_dist_sqr = max_dist * max_dist;
    if (cur_dist > max_dist_sqr) return;    
    int retval; 
    if (root->need_push_down_to_left || root->need_push_down_to_right) {
        retval = pthread_mutex_trylock(&(root->push_down_mutex_lock));
        if (retval == 0){
            Push_Down(root);
            pthread_mutex_unlock(&(root->push_down_mutex_lock));
        } else {
            pthread_mutex_lock(&(root->push_down_mutex_lock));
            pthread_mutex_unlock(&(root->push_down_mutex_lock));
        }
    }
    if (!root->point_deleted){
        float dist = calc_dist(point, root->point);
        if (dist <= max_dist_sqr && (q.size() < k_nearest || dist < q.top().dist)){
            if (q.size() >= k_nearest) q.pop();
            PointType_CMP current_point{root->point, dist};                    
            q.push(current_point);            
        }
    }  
    int cur_search_counter;
    float dist_left_node = calc_box_dist(root->left_son_ptr, point);
    float dist_right_node = calc_box_dist(root->right_son_ptr, point);
    if (q.size()< k_nearest || dist_left_node < q.top().dist && dist_right_node < q.top().dist){
        if (dist_left_node <= dist_right_node) {
            if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != root->left_son_ptr){
                Search(root->left_son_ptr, k_nearest, point, q, max_dist);                       
            } else {
                pthread_mutex_lock(&search_flag_mutex);
                while (search_mutex_counter == -1)
                {
                    pthread_mutex_unlock(&search_flag_mutex);
                    usleep(1);
                    pthread_mutex_lock(&search_flag_mutex);
                }
                search_mutex_counter += 1;
                pthread_mutex_unlock(&search_flag_mutex);
                Search(root->left_son_ptr, k_nearest, point, q, max_dist);  
                pthread_mutex_lock(&search_flag_mutex);
                search_mutex_counter -= 1;
                pthread_mutex_unlock(&search_flag_mutex);
            }
            if (q.size() < k_nearest || dist_right_node < q.top().dist) {
                if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != root->right_son_ptr){
                    Search(root->right_son_ptr, k_nearest, point, q, max_dist);                       
                } else {
                    pthread_mutex_lock(&search_flag_mutex);
                    while (search_mutex_counter == -1)
                    {
                        pthread_mutex_unlock(&search_flag_mutex);
                        usleep(1);
                        pthread_mutex_lock(&search_flag_mutex);
                    }
                    search_mutex_counter += 1;
                    pthread_mutex_unlock(&search_flag_mutex);                    
                    Search(root->right_son_ptr, k_nearest, point, q, max_dist);  
                    pthread_mutex_lock(&search_flag_mutex);
                    search_mutex_counter -= 1;
                    pthread_mutex_unlock(&search_flag_mutex);
                }                
            }
        } else {
            if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != root->right_son_ptr){
                Search(root->right_son_ptr, k_nearest, point, q, max_dist);                       
            } else {
                pthread_mutex_lock(&search_flag_mutex);
                while (search_mutex_counter == -1)
                {
                    pthread_mutex_unlock(&search_flag_mutex);
                    usleep(1);
                    pthread_mutex_lock(&search_flag_mutex);
                }
                search_mutex_counter += 1;
                pthread_mutex_unlock(&search_flag_mutex);                   
                Search(root->right_son_ptr, k_nearest, point, q, max_dist);  
                pthread_mutex_lock(&search_flag_mutex);
                search_mutex_counter -= 1;
                pthread_mutex_unlock(&search_flag_mutex);
            }
            if (q.size() < k_nearest || dist_left_node < q.top().dist) {            
                if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != root->left_son_ptr){
                    Search(root->left_son_ptr, k_nearest, point, q, max_dist);                       
                } else {
                    pthread_mutex_lock(&search_flag_mutex);
                    while (search_mutex_counter == -1)
                    {
                        pthread_mutex_unlock(&search_flag_mutex);
                        usleep(1);
                        pthread_mutex_lock(&search_flag_mutex);
                    }
                    search_mutex_counter += 1;
                    pthread_mutex_unlock(&search_flag_mutex);  
                    Search(root->left_son_ptr, k_nearest, point, q, max_dist);  
                    pthread_mutex_lock(&search_flag_mutex);
                    search_mutex_counter -= 1;
                    pthread_mutex_unlock(&search_flag_mutex);
                }
            }
        }
    } else {
        if (dist_left_node < q.top().dist) {        
            if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != root->left_son_ptr){
                Search(root->left_son_ptr, k_nearest, point, q, max_dist);                       
            } else {
                pthread_mutex_lock(&search_flag_mutex);
                while (search_mutex_counter == -1)
                {
                    pthread_mutex_unlock(&search_flag_mutex);
                    usleep(1);
                    pthread_mutex_lock(&search_flag_mutex);
                }
                search_mutex_counter += 1;
                pthread_mutex_unlock(&search_flag_mutex);  
                Search(root->left_son_ptr, k_nearest, point, q, max_dist);  
                pthread_mutex_lock(&search_flag_mutex);
                search_mutex_counter -= 1;
                pthread_mutex_unlock(&search_flag_mutex);
            }
        }
        if (dist_right_node < q.top().dist) {
            if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != root->right_son_ptr){
                Search(root->right_son_ptr, k_nearest, point, q, max_dist);                       
            } else {
                pthread_mutex_lock(&search_flag_mutex);
                while (search_mutex_counter == -1)
                {
                    pthread_mutex_unlock(&search_flag_mutex);
                    usleep(1);
                    pthread_mutex_lock(&search_flag_mutex);
                }
                search_mutex_counter += 1;
                pthread_mutex_unlock(&search_flag_mutex);  
                Search(root->right_son_ptr, k_nearest, point, q, max_dist);
                pthread_mutex_lock(&search_flag_mutex);
                search_mutex_counter -= 1;
                pthread_mutex_unlock(&search_flag_mutex);
            }
        }
    }
    return;
}
// 工具属性，从根节点开始向下递归搜索，获得所有被Box包含的节点。
// 前序遍历搜索，根据给定的box，从树中搜索被Box包围的点集
template <typename PointType>
void KD_TREE<PointType>::Search_by_range(KD_TREE_NODE *root, BoxPointType boxpoint, PointVector & Storage){
    if (root == nullptr) return;
     // 现将当前节点的属性Push down一下
    Push_Down(root);       
    // 当且仅当两个空间有交叉时，才有继续的必要。
    // 前三行是判断给定的Box范围是否与树有交集，没有的话就不用搜索了
    if (boxpoint.vertex_max[0] <= root->node_range_x[0] || boxpoint.vertex_min[0] > root->node_range_x[1]) return;
    if (boxpoint.vertex_max[1] <= root->node_range_y[0] || boxpoint.vertex_min[1] > root->node_range_y[1]) return;
    if (boxpoint.vertex_max[2] <= root->node_range_z[0] || boxpoint.vertex_min[2] > root->node_range_z[1]) return;
    // 给定的box完全包围当前的树，则将当前树展平，将所有点放到Storage内，直接返回即可
    if (boxpoint.vertex_min[0] <= root->node_range_x[0] && boxpoint.vertex_max[0] > root->node_range_x[1] && boxpoint.vertex_min[1] <= root->node_range_y[0] && boxpoint.vertex_max[1] > root->node_range_y[1] && boxpoint.vertex_min[2] <= root->node_range_z[0] && boxpoint.vertex_max[2] > root->node_range_z[1]){
        flatten(root, Storage, NOT_RECORD);
        return;
    }
     // 如果当前节点的point被Box包含，记录该point。
    // 给定的box完全包围当前节点，如果该点没有被删除，则将该点压入Storage
    if (boxpoint.vertex_min[0] <= root->point.x && boxpoint.vertex_max[0] > root->point.x && boxpoint.vertex_min[1] <= root->point.y && boxpoint.vertex_max[1] > root->point.y && boxpoint.vertex_min[2] <= root->point.z && boxpoint.vertex_max[2] > root->point.z){
        if (!root->point_deleted) Storage.push_back(root->point);
    } // 到左右子树中递归搜索，直到叶子节点结束
    if ((Rebuild_Ptr == nullptr) || root->left_son_ptr != *Rebuild_Ptr){
        Search_by_range(root->left_son_ptr, boxpoint, Storage);
    } else {
        pthread_mutex_lock(&search_flag_mutex);
        Search_by_range(root->left_son_ptr, boxpoint, Storage);
        pthread_mutex_unlock(&search_flag_mutex);
    }
    if ((Rebuild_Ptr == nullptr) || root->right_son_ptr != *Rebuild_Ptr){
        Search_by_range(root->right_son_ptr, boxpoint, Storage);
    } else {
        pthread_mutex_lock(&search_flag_mutex);
        Search_by_range(root->right_son_ptr, boxpoint, Storage);
        pthread_mutex_unlock(&search_flag_mutex);
    }
    return;    
}

template <typename PointType>
void KD_TREE<PointType>::Search_by_radius(KD_TREE_NODE *root, PointType point, float radius, PointVector &Storage)
{
    if (root == nullptr)
        return;
    Push_Down(root);
    PointType range_center;
    range_center.x = (root->node_range_x[0] + root->node_range_x[1]) * 0.5;
    range_center.y = (root->node_range_y[0] + root->node_range_y[1]) * 0.5;
    range_center.z = (root->node_range_z[0] + root->node_range_z[1]) * 0.5;
    float dist = sqrt(calc_dist(range_center, point));
    if (dist > radius + sqrt(root->radius_sq)) return;
    if (dist <= radius - sqrt(root->radius_sq)) 
    {
        flatten(root, Storage, NOT_RECORD);
        return;
    }
    if (!root->point_deleted && calc_dist(root->point, point) <= radius * radius){
        Storage.push_back(root->point);
    }
    if ((Rebuild_Ptr == nullptr) || root->left_son_ptr != *Rebuild_Ptr)
    {
        Search_by_radius(root->left_son_ptr, point, radius, Storage);
    }
    else
    {
        pthread_mutex_lock(&search_flag_mutex);
        Search_by_radius(root->left_son_ptr, point, radius, Storage);
        pthread_mutex_unlock(&search_flag_mutex);
    }
    if ((Rebuild_Ptr == nullptr) || root->right_son_ptr != *Rebuild_Ptr)
    {
        Search_by_radius(root->right_son_ptr, point, radius, Storage);
    }
    else
    {
        pthread_mutex_lock(&search_flag_mutex);
        Search_by_radius(root->right_son_ptr, point, radius, Storage);
        pthread_mutex_unlock(&search_flag_mutex);
    }    
    return;
}

template <typename PointType>
bool KD_TREE<PointType>::Criterion_Check(KD_TREE_NODE * root){
    // 如果当前树的总节点个数小于最小的树的size，则没必要重建，返回false
    if (root->TreeSize <= Minimal_Unbalanced_Tree_Size){
        return false;
    }
    // 平衡性归零
    float balance_evaluation = 0.0f;
    float delete_evaluation = 0.0f;
    // 新建一个节点指针存储树的根节点，root是虚拟头结点
    KD_TREE_NODE * son_ptr = root->left_son_ptr;
    if (son_ptr == nullptr) son_ptr = root->right_son_ptr;
    // 无效点在树中所占的比例
    delete_evaluation = float(root->invalid_point_num)/ root->TreeSize;
    // 有效点在树中所占的比例
    balance_evaluation = float(son_ptr->TreeSize) / (root->TreeSize-1);  
    if (delete_evaluation > delete_criterion_param){// 论文中的平衡性判断准则
        return true;
    }
    if (balance_evaluation > balance_criterion_param || balance_evaluation < 1-balance_criterion_param){
        return true;
    } 
    return false;
}
// 当属性Pushdown 为true时，Pushdown 函数将被deleted, treedeleted 和 pushdown的标签复制到它的子节点（但没有复制到它的的后代。
template <typename PointType>
void KD_TREE<PointType>::Push_Down(KD_TREE_NODE *root){
    if (root == nullptr) return;
    Operation_Logger_Type operation;// 实例化一个操作记录器
    operation.op = PUSH_DOWN;// 操作器设置为PUSH_DOWN状态
    operation.tree_deleted = root->tree_deleted;// 获取当前节点的删除标签，用于传递给子节点
    operation.tree_downsample_deleted = root->tree_downsample_deleted;// 同样，获取当前节点的降采样删除标签，用于传递给子节点
    // 左子节点不为空，且该节点需要push down
    if (root->need_push_down_to_left && root->left_son_ptr != nullptr){
        // 当前不需要重建 或 需要重建的树不是左子树
        if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != root->left_son_ptr){
            // 将当前节点的删除标签传递给左子节点
            root->left_son_ptr->tree_downsample_deleted |= root->tree_downsample_deleted;
            root->left_son_ptr->point_downsample_deleted |= root->tree_downsample_deleted;
            root->left_son_ptr->tree_deleted = root->tree_deleted || root->left_son_ptr->tree_downsample_deleted;
            root->left_son_ptr->point_deleted = root->left_son_ptr->tree_deleted || root->left_son_ptr->point_downsample_deleted;
            if (root->tree_downsample_deleted) root->left_son_ptr->down_del_num = root->left_son_ptr->TreeSize;
            if (root->tree_deleted) root->left_son_ptr->invalid_point_num = root->left_son_ptr->TreeSize;
                else root->left_son_ptr->invalid_point_num = root->left_son_ptr->down_del_num;
            // 标记下左子节点的状态更新情况，设为true意味着左子节点更新完，它的属性也要传递给后代节点
            root->left_son_ptr->need_push_down_to_left = true;
            root->left_son_ptr->need_push_down_to_right = true;
            // 当前节点属性已经Push down了，所以可以将其更新push down需求设为false
            root->need_push_down_to_left = false;                
        } else {// if条件的逆否命题：当前需要重建 并且 需要重建的树是左子树
            pthread_mutex_lock(&working_flag_mutex);
            // 将当前节点的删除标签传递给左子节点
            root->left_son_ptr->tree_downsample_deleted |= root->tree_downsample_deleted;
            root->left_son_ptr->point_downsample_deleted |= root->tree_downsample_deleted;
            root->left_son_ptr->tree_deleted = root->tree_deleted || root->left_son_ptr->tree_downsample_deleted;
            root->left_son_ptr->point_deleted = root->left_son_ptr->tree_deleted || root->left_son_ptr->point_downsample_deleted;
            if (root->tree_downsample_deleted) root->left_son_ptr->down_del_num = root->left_son_ptr->TreeSize;
            if (root->tree_deleted) root->left_son_ptr->invalid_point_num = root->left_son_ptr->TreeSize;
                else root->left_son_ptr->invalid_point_num = root->left_son_ptr->down_del_num;  
            // 标记下左子节点的状态更新情况，设为true意味着左子节点更新完，它的属性也要传递给后代节点          
            root->left_son_ptr->need_push_down_to_left = true;
            root->left_son_ptr->need_push_down_to_right = true;
            // 将当前节点需要push down给左子树根节点的操作记录到Rebuild_Logger，等重建完成后补作业
            if (rebuild_flag){
                pthread_mutex_lock(&rebuild_logger_mutex_lock);
                Rebuild_Logger.push(operation);
                pthread_mutex_unlock(&rebuild_logger_mutex_lock);
            }
            root->need_push_down_to_left = false;
            pthread_mutex_unlock(&working_flag_mutex);            
        }
    }// 对于右子节点也是相同的Push down操作，传递当前节点的属性下去
    if (root->need_push_down_to_right && root->right_son_ptr != nullptr){
        if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != root->right_son_ptr){
            root->right_son_ptr->tree_downsample_deleted |= root->tree_downsample_deleted;
            root->right_son_ptr->point_downsample_deleted |= root->tree_downsample_deleted;
            root->right_son_ptr->tree_deleted = root->tree_deleted || root->right_son_ptr->tree_downsample_deleted;
            root->right_son_ptr->point_deleted = root->right_son_ptr->tree_deleted || root->right_son_ptr->point_downsample_deleted;
            if (root->tree_downsample_deleted) root->right_son_ptr->down_del_num = root->right_son_ptr->TreeSize;
            if (root->tree_deleted) root->right_son_ptr->invalid_point_num = root->right_son_ptr->TreeSize;
                else root->right_son_ptr->invalid_point_num = root->right_son_ptr->down_del_num;
            root->right_son_ptr->need_push_down_to_left = true;
            root->right_son_ptr->need_push_down_to_right = true;
            root->need_push_down_to_right = false;
        } else {
            pthread_mutex_lock(&working_flag_mutex);
            root->right_son_ptr->tree_downsample_deleted |= root->tree_downsample_deleted;
            root->right_son_ptr->point_downsample_deleted |= root->tree_downsample_deleted;
            root->right_son_ptr->tree_deleted = root->tree_deleted || root->right_son_ptr->tree_downsample_deleted;
            root->right_son_ptr->point_deleted = root->right_son_ptr->tree_deleted || root->right_son_ptr->point_downsample_deleted;
            if (root->tree_downsample_deleted) root->right_son_ptr->down_del_num = root->right_son_ptr->TreeSize;
            if (root->tree_deleted) root->right_son_ptr->invalid_point_num = root->right_son_ptr->TreeSize;
                else root->right_son_ptr->invalid_point_num = root->right_son_ptr->down_del_num;            
            root->right_son_ptr->need_push_down_to_left = true;
            root->right_son_ptr->need_push_down_to_right = true;
            if (rebuild_flag){
                pthread_mutex_lock(&rebuild_logger_mutex_lock);
                Rebuild_Logger.push(operation);
                pthread_mutex_unlock(&rebuild_logger_mutex_lock);
            }            
            root->need_push_down_to_right = false;
            pthread_mutex_unlock(&working_flag_mutex);
        }
    }
    return;
}
// Update函数的主要目的是为了更新该节点以及它子树的点的范围以及size，还有他的del，bal，
// 用来判断树是否平衡，是否需要重建。这个函数在原论文中是pull-up操作（树的属性由子向父传播）。
 
// Pullup函数将基于T的子树的信息汇总给T的以下属性：treesize（见数据结构1，第5行）保存子树上
// 所有节点数，invalidnum 存储子树上标记为“删除”的节点数，range（见数据结构1，第7行）汇总子
// 树上所有点坐标轴的范围，其中k为点维度。
template <typename PointType>
void KD_TREE<PointType>::Update(KD_TREE_NODE * root){// 该函数不需要递归，因为父节点的属性只与左右子节点相关
    KD_TREE_NODE * left_son_ptr = root->left_son_ptr;
    KD_TREE_NODE * right_son_ptr = root->right_son_ptr;
    float tmp_range_x[2] = {INFINITY, -INFINITY};
    float tmp_range_y[2] = {INFINITY, -INFINITY};
    float tmp_range_z[2] = {INFINITY, -INFINITY};
    // Update Tree Size   
    if (left_son_ptr != nullptr && right_son_ptr != nullptr){//两边不为空
        root->TreeSize = left_son_ptr->TreeSize + right_son_ptr->TreeSize + 1;// 当前树的节点数=左子树节点数+右子树节点数+1(当前节点本身)
        root->invalid_point_num = left_son_ptr->invalid_point_num + right_son_ptr->invalid_point_num + (root->point_deleted? 1:0);// 同上，树中的无效点个数也类似计算
        root->down_del_num = left_son_ptr->down_del_num + right_son_ptr->down_del_num + (root->point_downsample_deleted? 1:0);// 降采样删除的节点个数和
        root->tree_downsample_deleted = left_son_ptr->tree_downsample_deleted & right_son_ptr->tree_downsample_deleted & root->point_downsample_deleted;// 降采样删除的标志位
        root->tree_deleted = left_son_ptr->tree_deleted && right_son_ptr->tree_deleted && root->point_deleted;// 删除树的标志位
        if (root->tree_deleted || (!left_son_ptr->tree_deleted && !right_son_ptr->tree_deleted && !root->point_deleted)){//删除整个树或者全都不删
            tmp_range_x[0] = min(min(left_son_ptr->node_range_x[0],right_son_ptr->node_range_x[0]),root->point.x);// 获取包围当前树的box
            tmp_range_x[1] = max(max(left_son_ptr->node_range_x[1],right_son_ptr->node_range_x[1]),root->point.x);
            tmp_range_y[0] = min(min(left_son_ptr->node_range_y[0],right_son_ptr->node_range_y[0]),root->point.y);
            tmp_range_y[1] = max(max(left_son_ptr->node_range_y[1],right_son_ptr->node_range_y[1]),root->point.y);
            tmp_range_z[0] = min(min(left_son_ptr->node_range_z[0],right_son_ptr->node_range_z[0]),root->point.z);
            tmp_range_z[1] = max(max(left_son_ptr->node_range_z[1],right_son_ptr->node_range_z[1]),root->point.z);
        } else { // 否则的话，肯定有一边子树被标记了删除了
            if (!left_son_ptr->tree_deleted){// 仅取左子树
                tmp_range_x[0] = min(tmp_range_x[0], left_son_ptr->node_range_x[0]);
                tmp_range_x[1] = max(tmp_range_x[1], left_son_ptr->node_range_x[1]);
                tmp_range_y[0] = min(tmp_range_y[0], left_son_ptr->node_range_y[0]);
                tmp_range_y[1] = max(tmp_range_y[1], left_son_ptr->node_range_y[1]);
                tmp_range_z[0] = min(tmp_range_z[0], left_son_ptr->node_range_z[0]);
                tmp_range_z[1] = max(tmp_range_z[1], left_son_ptr->node_range_z[1]);
            }
            if (!right_son_ptr->tree_deleted){
                tmp_range_x[0] = min(tmp_range_x[0], right_son_ptr->node_range_x[0]);
                tmp_range_x[1] = max(tmp_range_x[1], right_son_ptr->node_range_x[1]);
                tmp_range_y[0] = min(tmp_range_y[0], right_son_ptr->node_range_y[0]);
                tmp_range_y[1] = max(tmp_range_y[1], right_son_ptr->node_range_y[1]);
                tmp_range_z[0] = min(tmp_range_z[0], right_son_ptr->node_range_z[0]);
                tmp_range_z[1] = max(tmp_range_z[1], right_son_ptr->node_range_z[1]);                
            }
            if (!root->point_deleted){// 子树的box+root的点
                tmp_range_x[0] = min(tmp_range_x[0], root->point.x);
                tmp_range_x[1] = max(tmp_range_x[1], root->point.x);
                tmp_range_y[0] = min(tmp_range_y[0], root->point.y);
                tmp_range_y[1] = max(tmp_range_y[1], root->point.y);
                tmp_range_z[0] = min(tmp_range_z[0], root->point.z);
                tmp_range_z[1] = max(tmp_range_z[1], root->point.z);                 
            }
        }
    } else if (left_son_ptr != nullptr){ // 仅有左子树，右子树为空
        root->TreeSize = left_son_ptr->TreeSize + 1;// 仅需要将左子树的信息汇总给root
        root->invalid_point_num = left_son_ptr->invalid_point_num + (root->point_deleted?1:0);
        root->down_del_num = left_son_ptr->down_del_num + (root->point_downsample_deleted?1:0);
        root->tree_downsample_deleted = left_son_ptr->tree_downsample_deleted & root->point_downsample_deleted;
        root->tree_deleted = left_son_ptr->tree_deleted && root->point_deleted;
        if (root->tree_deleted || (!left_son_ptr->tree_deleted && !root->point_deleted)){// 获取包围左子树+root的box
            tmp_range_x[0] = min(left_son_ptr->node_range_x[0],root->point.x);
            tmp_range_x[1] = max(left_son_ptr->node_range_x[1],root->point.x);
            tmp_range_y[0] = min(left_son_ptr->node_range_y[0],root->point.y);
            tmp_range_y[1] = max(left_son_ptr->node_range_y[1],root->point.y); 
            tmp_range_z[0] = min(left_son_ptr->node_range_z[0],root->point.z);
            tmp_range_z[1] = max(left_son_ptr->node_range_z[1],root->point.z);  
        } else {
            if (!left_son_ptr->tree_deleted){
                tmp_range_x[0] = min(tmp_range_x[0], left_son_ptr->node_range_x[0]);
                tmp_range_x[1] = max(tmp_range_x[1], left_son_ptr->node_range_x[1]);
                tmp_range_y[0] = min(tmp_range_y[0], left_son_ptr->node_range_y[0]);
                tmp_range_y[1] = max(tmp_range_y[1], left_son_ptr->node_range_y[1]);
                tmp_range_z[0] = min(tmp_range_z[0], left_son_ptr->node_range_z[0]);
                tmp_range_z[1] = max(tmp_range_z[1], left_son_ptr->node_range_z[1]);                
            }
            if (!root->point_deleted){
                tmp_range_x[0] = min(tmp_range_x[0], root->point.x);
                tmp_range_x[1] = max(tmp_range_x[1], root->point.x);
                tmp_range_y[0] = min(tmp_range_y[0], root->point.y);
                tmp_range_y[1] = max(tmp_range_y[1], root->point.y);
                tmp_range_z[0] = min(tmp_range_z[0], root->point.z);
                tmp_range_z[1] = max(tmp_range_z[1], root->point.z);                 
            }            
        }

    } else if (right_son_ptr != nullptr){// 仅有右子树，左子树为空
        root->TreeSize = right_son_ptr->TreeSize + 1;
        root->invalid_point_num = right_son_ptr->invalid_point_num + (root->point_deleted? 1:0);
        root->down_del_num = right_son_ptr->down_del_num + (root->point_downsample_deleted? 1:0);        
        root->tree_downsample_deleted = right_son_ptr->tree_downsample_deleted & root->point_downsample_deleted;
        root->tree_deleted = right_son_ptr->tree_deleted && root->point_deleted;
        if (root->tree_deleted || (!right_son_ptr->tree_deleted && !root->point_deleted)){
            tmp_range_x[0] = min(right_son_ptr->node_range_x[0],root->point.x);
            tmp_range_x[1] = max(right_son_ptr->node_range_x[1],root->point.x);
            tmp_range_y[0] = min(right_son_ptr->node_range_y[0],root->point.y);
            tmp_range_y[1] = max(right_son_ptr->node_range_y[1],root->point.y); 
            tmp_range_z[0] = min(right_son_ptr->node_range_z[0],root->point.z);
            tmp_range_z[1] = max(right_son_ptr->node_range_z[1],root->point.z); 
        } else {
            if (!right_son_ptr->tree_deleted){
                tmp_range_x[0] = min(tmp_range_x[0], right_son_ptr->node_range_x[0]);
                tmp_range_x[1] = max(tmp_range_x[1], right_son_ptr->node_range_x[1]);
                tmp_range_y[0] = min(tmp_range_y[0], right_son_ptr->node_range_y[0]);
                tmp_range_y[1] = max(tmp_range_y[1], right_son_ptr->node_range_y[1]);
                tmp_range_z[0] = min(tmp_range_z[0], right_son_ptr->node_range_z[0]);
                tmp_range_z[1] = max(tmp_range_z[1], right_son_ptr->node_range_z[1]);                
            }
            if (!root->point_deleted){
                tmp_range_x[0] = min(tmp_range_x[0], root->point.x);
                tmp_range_x[1] = max(tmp_range_x[1], root->point.x);
                tmp_range_y[0] = min(tmp_range_y[0], root->point.y);
                tmp_range_y[1] = max(tmp_range_y[1], root->point.y);
                tmp_range_z[0] = min(tmp_range_z[0], root->point.z);
                tmp_range_z[1] = max(tmp_range_z[1], root->point.z);                 
            }            
        }
    } else {// 左右子树都为空
        root->TreeSize = 1;
        root->invalid_point_num = (root->point_deleted? 1:0);
        root->down_del_num = (root->point_downsample_deleted? 1:0);
        root->tree_downsample_deleted = root->point_downsample_deleted;
        root->tree_deleted = root->point_deleted;
        tmp_range_x[0] = root->point.x;// box收缩成一个点
        tmp_range_x[1] = root->point.x;        
        tmp_range_y[0] = root->point.y;
        tmp_range_y[1] = root->point.y; 
        tmp_range_z[0] = root->point.z;
        tmp_range_z[1] = root->point.z;                 
    }
    // 用memcpy函数直接拷贝内存空间，节约时间花销 等同于 node_range=tmp_range
    // 将树的区间范围拷贝给节点属性 node_range_x = [node_range_x_min, node_range_x_max]
    memcpy(root->node_range_x,tmp_range_x,sizeof(tmp_range_x));
    memcpy(root->node_range_y,tmp_range_y,sizeof(tmp_range_y));
    memcpy(root->node_range_z,tmp_range_z,sizeof(tmp_range_z));
    float x_L = (root->node_range_x[1] - root->node_range_x[0]) * 0.5;//节点坐标范围*0.5
    float y_L = (root->node_range_y[1] - root->node_range_y[0]) * 0.5;
    float z_L = (root->node_range_z[1] - root->node_range_z[0]) * 0.5;
    root->radius_sq = x_L*x_L + y_L * y_L + z_L * z_L;    
    // 让子节点知道自身父节点是谁
    if (left_son_ptr != nullptr) left_son_ptr -> father_ptr = root;
    if (right_son_ptr != nullptr) right_son_ptr -> father_ptr = root;
    // 如果当前节点为整个树的实际根节点，且树的节点数大于3
    if (root == Root_Node && root->TreeSize > 3){
        KD_TREE_NODE * son_ptr = root->left_son_ptr;// 获取其左子树
        if (son_ptr == nullptr) son_ptr = root->right_son_ptr;// 左子树为null则获取右子树
        // 计算子树节点个数和整个树的节点个数比例用于判断平衡性
        float tmp_bal = float(son_ptr->TreeSize) / (root->TreeSize-1);
        root->alpha_del = float(root->invalid_point_num)/ root->TreeSize;//无效点/总数
        root->alpha_bal = (tmp_bal>=0.5-EPSS)?tmp_bal:1-tmp_bal;
        //子树节点和剩余部分占比更大的那个比例？
    }   
    return;
}
// 前序遍历和后序遍历都有，将一个树的所有节点展平，按顺序放到一个容器内
template <typename PointType>
void KD_TREE<PointType>::flatten(KD_TREE_NODE * root, PointVector &Storage, delete_point_storage_set storage_type){
    if (root == nullptr) return;
    Push_Down(root);// 将root的属性传递给它的左右子节点
    if (!root->point_deleted) {// 只要该点没有被删除，那么就把它记录下来， 前序遍历
        Storage.push_back(root->point);
    }
    flatten(root->left_son_ptr, Storage, storage_type);
    flatten(root->right_son_ptr, Storage, storage_type);
    switch (storage_type)// 根据枚举变量storage_type判断使用什么存储动作， 后序遍历
    {
    case NOT_RECORD: // 不操作
        break;
    case DELETE_POINTS_REC:// 删除点的记录
        // 只有该点是真正被删除，而不是被降采样删除，该点才会被记录
        if (root->point_deleted && !root->point_downsample_deleted) {
            Points_deleted.push_back(root->point);
        }       
        break;
    case MULTI_THREAD_REC: // 多线程重建树时的点的记录
        if (root->point_deleted  && !root->point_downsample_deleted) {
            Multithread_Points_deleted.push_back(root->point);
        }
        break;
    default:
        break;
    }     
    return;
}

template <typename PointType>
void KD_TREE<PointType>::delete_tree_nodes(KD_TREE_NODE ** root){ 
    if (*root == nullptr) return;
    Push_Down(*root);    
    delete_tree_nodes(&(*root)->left_son_ptr);
    delete_tree_nodes(&(*root)->right_son_ptr);
    
    pthread_mutex_destroy( &(*root)->push_down_mutex_lock);         
    delete *root;
    *root = nullptr;                    

    return;
}

template <typename PointType>
bool KD_TREE<PointType>::same_point(PointType a, PointType b){
    return (fabs(a.x-b.x) < EPSS && fabs(a.y-b.y) < EPSS && fabs(a.z-b.z) < EPSS );
}

template <typename PointType>
float KD_TREE<PointType>::calc_dist(PointType a, PointType b){
    float dist = 0.0f;
    dist = (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z);
    return dist;
}

template <typename PointType>
float KD_TREE<PointType>::calc_box_dist(KD_TREE_NODE * node, PointType point){
    if (node == nullptr) return INFINITY;
    float min_dist = 0.0;
    if (point.x < node->node_range_x[0]) min_dist += (point.x - node->node_range_x[0])*(point.x - node->node_range_x[0]);
    if (point.x > node->node_range_x[1]) min_dist += (point.x - node->node_range_x[1])*(point.x - node->node_range_x[1]);
    if (point.y < node->node_range_y[0]) min_dist += (point.y - node->node_range_y[0])*(point.y - node->node_range_y[0]);
    if (point.y > node->node_range_y[1]) min_dist += (point.y - node->node_range_y[1])*(point.y - node->node_range_y[1]);
    if (point.z < node->node_range_z[0]) min_dist += (point.z - node->node_range_z[0])*(point.z - node->node_range_z[0]);
    if (point.z > node->node_range_z[1]) min_dist += (point.z - node->node_range_z[1])*(point.z - node->node_range_z[1]);
    return min_dist;
}

template <typename PointType> bool KD_TREE<PointType>::point_cmp_x(PointType a, PointType b) { return a.x < b.x;}
template <typename PointType> bool KD_TREE<PointType>::point_cmp_y(PointType a, PointType b) { return a.y < b.y;}
template <typename PointType> bool KD_TREE<PointType>::point_cmp_z(PointType a, PointType b) { return a.z < b.z;}

// manual queue
template <typename T>
void MANUAL_Q<T>::clear(){
    head = 0;
    tail = 0;
    counter = 0;
    is_empty = true;
    return;
}

template <typename T>
void MANUAL_Q<T>::pop(){
    if (counter == 0) return;
    head ++;
    head %= Q_LEN;
    counter --;
    if (counter == 0) is_empty = true;
    return;
}

template <typename T>
T MANUAL_Q<T>::front(){
    return q[head];
}

template <typename T>
T MANUAL_Q<T>::back(){
    return q[tail];
}

template <typename T>
void MANUAL_Q<T>::push(T op){
    q[tail] = op;
    counter ++;
    if (is_empty) is_empty = false;
    tail ++;
    tail %= Q_LEN;
}

template <typename T>
bool MANUAL_Q<T>::empty(){
    return is_empty;
}

template <typename T>
int MANUAL_Q<T>::size(){
    return counter;
}

template class KD_TREE<ikdTree_PointType>;
template class KD_TREE<pcl::PointXYZ>;
template class KD_TREE<pcl::PointXYZI>;
template class KD_TREE<pcl::PointXYZINormal>;